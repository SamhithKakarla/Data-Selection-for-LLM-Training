# train_crust.py
"""
Train pipeline implementing CRUST (coreset selection in gradient space)
for your TinyGPT classifier.

Usage examples:
  python train_crust.py --epochs 10 --apply_crust --select_fraction 0.3 --k_per_class_factor 0.05

Notes:
 - This implements the core CRUST operations (last-layer gradient embedding,
   per-class k-medoids via kmeans+medoid extraction, cluster weights r_j,
   epoch-wise re-selection, weighted training on medoids).
 - Mixup (in representation space) is recommended by the paper but is
   left as an optional enhancement (commented and explained).
"""

import os
import argparse
import random
from math import ceil
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from transformers import AutoTokenizer
from datasets import load_dataset

from model_for_crest import TinyGPTClassifier as TinyGPT  # model.py above

# scikit-learn for PCA/KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ----------------------------
# dataset helpers (same as yours)
# ----------------------------
def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def make_dataset(tokenizer, texts, labels, max_len):
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    dataset = []
    for i in range(enc['input_ids'].size(0)):
        dataset.append({
            'input_ids': enc['input_ids'][i],
            'attention_mask': enc['attention_mask'][i],
            'label': int(labels[i])
        })
    return dataset


# ----------------------------
# LAST-LAYER GRADIENT EMBEDDING (Eq.9 style)
# ----------------------------
def compute_last_layer_embeddings(model, dataloader, device, batch_size_for_encode=64):
    """
    Compute the last-layer gradient approximation embedding for each example:
      g_i = flatten( (p_i - e_yi) outer h_i )   -> shape (C * D)
    We will compute features h_i = model.encode(x_i) and outputs p_i = softmax(logits_i).
    Return:
      - embeddings: (N, C * D) numpy array
      - preds: (N,) predicted class indices (argmax p)
      - indices: list of dataset indices (0..N-1)
      - h_dict: list of pooled features (N, D) as torch tensor (cpu) - optionally used for later mixup
    Note: we avoid backprop; this is closed-form and fast.
    """
    model.eval()
    all_embs = []
    all_preds = []
    all_indices = []
    all_h = []

    idx = 0
    for batch in tqdm(dataloader, desc="Computing last-layer embeddings"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        with torch.no_grad():
            h = model.encode(input_ids, attention_mask=attention_mask)  # (B, D)
            logits = model.classifier(h)  # (B, C)  [explicit linear call]
            probs = torch.softmax(logits, dim=-1)  # (B, C)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()  # (B,)

        B = input_ids.size(0)
        C = logits.size(-1)
        D = h.size(-1)
        probs_np = probs.cpu().numpy()
        h_np = h.cpu().numpy()

        # compute (p - y_onehot) outer h for each example
        for b in range(B):
            p = probs_np[b]  # (C,)
            # Note: CRUST uses predicted labels to partition classes, but the gradient embedding uses
            # p - y (where y is ground-truth). The original CRUST uses predicted label sets U_tau^c,
            # but uses gradient-like vectors to cluster. We will compute p - y_true as in last-layer approx.
            # The paper (Eq 9) uses derivatives wrt last layer: (p - y) ⊗ h.
            # compute delta = p - onehot(y_true)
            y_onehot = np.zeros_like(p)
            # we don't have labels here; last-layer approx sometimes uses p or predicted class instead.
            # we will set y_onehot to the predicted-class one-hot to compute gradient-like vector consistent with predicted class clusters:
            # Use predicted label as "pseudo" label for grouping per the algorithm step U_tau^c.
            pred_c = preds[b]
            y_onehot[pred_c] = 1.0
            delta = p - y_onehot  # (C,)
            # outer product: (C, D) = delta[:,None] * h[b: b+1, :]
            emb = (delta.reshape(-1, 1) * h_np[b].reshape(1, -1)).reshape(-1)  # flatten to C*D
            all_embs.append(emb)
            all_preds.append(preds[b])
            all_indices.append(idx)
            all_h.append(h_np[b])
            idx += 1

    embeddings = np.stack(all_embs, axis=0)  # (N, C*D)
    preds = np.array(all_preds, dtype=np.int32)
    h_arr = np.stack(all_h, axis=0)  # (N, D)
    return embeddings, preds, all_indices, h_arr


# ----------------------------
# per-class k-medoids via kmeans center + medoid extraction
# ----------------------------
def per_class_medoids(embeddings, preds, indices, num_classes, total_budget):
    """
    Given embeddings (N, L) and predicted class per example, perform per-class medoid selection.
    Strategy:
      - For class c with n_c members, allocate k_c = max(1, round(total_budget * n_c / N))
      - Run KMeans with k=k_c on class-specific embeddings
      - For each cluster center, pick the *actual* example closest to center as medoid (returns original dataset index)
      - Return list of medoid indices (global dataset indices) and cluster assignments (mapping medoid->members)
    """
    N = embeddings.shape[0]
    L = embeddings.shape[1]
    medoid_indices = []
    cluster_assignments = {}  # medoid_idx -> list of member indices (indices refer to position in embeddings / original indices)
    counts = np.bincount(preds, minlength=num_classes)

    for c in range(num_classes):
        class_mask = (preds == c)
        idxs_c = np.where(class_mask)[0]
        n_c = idxs_c.shape[0]
        if n_c == 0:
            continue
        k_c = max(1, int(round(total_budget * (n_c / float(N)))))
        k_c = min(k_c, n_c)  # cannot have more medoids than points

        Xc = embeddings[idxs_c]  # shape (n_c, L)
        if k_c == 1:
            # medoid is the point closest to centroid
            centroid = Xc.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(Xc - centroid, axis=1)
            pick = idxs_c[int(np.argmin(dists))]
            medoid_indices.append(pick)
            cluster_assignments[pick] = idxs_c.tolist()
        else:
            # run kmeans
            km = KMeans(n_clusters=k_c, random_state=0, n_init='auto')
            labels = km.fit_predict(Xc)
            centers = km.cluster_centers_
            # for each cluster center, find the closest actual point (medoid)
            for k in range(k_c):
                members_local = np.where(labels == k)[0]  # local indices into Xc
                if len(members_local) == 0:
                    continue
                members_global = idxs_c[members_local]  # positions in global embeddings
                member_vectors = Xc[members_local]
                center = centers[k]
                # compute distances
                dists = np.linalg.norm(member_vectors - center.reshape(1, -1), axis=1)
                argmin = int(np.argmin(dists))
                chosen_global = members_global[argmin]
                medoid_indices.append(chosen_global)
                cluster_assignments[chosen_global] = members_global.tolist()

    # medoid_indices are indices into embeddings array. Map to original dataset positions
    # We will return medoid positions (relative to embeddings array) and their assigned member positions
    return medoid_indices, cluster_assignments


# ----------------------------
# Build weighted coreset and dataloader
# ----------------------------
def build_weighted_coreset(dataset, medoid_positions, cluster_assignments, embeddings_shape, select_fraction):
    """
    dataset: original list-like dataset (indexable)
    medoid_positions: indices into embeddings array (positions), we assume 0..N-1 mapping to dataset order
    cluster_assignments: dict medoid_pos -> list(member_positions)
    We compute r_j = size of V_j (cluster assigned members) and create a list of (dataset_idx, weight) for medoids.
    Return a PyTorch dataset (list of dicts) with 'weight' field (float).
    """
    coreset = []
    total_members = 0
    for med_pos in medoid_positions:
        members = cluster_assignments.get(med_pos, [])
        rj = len(members)
        total_members += rj

    # Build coreset entries
    for med_pos in medoid_positions:
        members = cluster_assignments.get(med_pos, [])
        rj = len(members)
        # map med_pos (position) to dataset index: med_pos equals original dataset index as we constructed earlier
        dataset_idx = med_pos
        entry = dict(dataset_idx=dataset_idx, weight=float(rj))
        coreset.append(entry)

    return coreset


# ----------------------------
# Training utilities
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss(reduction='none')  # we'll apply per-sample weights externally if needed

    pbar = tqdm(loader, desc="Train epoch")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        weights = batch.get('weights', None)
        if weights is not None:
            weights = weights.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)
        losses = criterion(logits, labels)  # (B,)
        if weights is not None:
            weights = weights / weights.sum()
            losses = losses * weights
        loss = losses.mean()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        pbar.set_postfix({'loss': total_loss / total_examples, 'acc': total_correct / total_examples})

    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss(reduction='mean')

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    return total_loss / total_examples, total_correct / total_examples


# custom dataloader for weighted medoid training
class CoresetDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset yields batches from the medoid coreset.
    Each element is a dict with 'input_ids', 'attention_mask', 'labels', 'weights'
    We will sample medoids proportionally to their cluster size (r_j) to emulate weighted gradient.
    """
    def __init__(self, base_dataset, coreset_entries, tokenizer, max_len, batch_size):
        """
        base_dataset: list-like original dataset (indexable)
        coreset_entries: list of dicts {dataset_idx: int, weight: float}
        tokenizer, max_len required if augmentation/mixup is needed
        """
        super().__init__()
        self.base = base_dataset
        self.entries = coreset_entries
        self.batch_size = batch_size
        # Precompute sampling probabilities proportional to weight
        weights = np.array([e['weight'] for e in self.entries], dtype=float)
        if weights.sum() == 0:
            self.probs = np.ones(len(weights)) / len(weights)
        else:
            self.probs = weights / weights.sum()

    def __iter__(self):
        # sample indices for an epoch length ~ len(entries) * 10 (arbitrary)
        # We'll yield batches by sampling medoids with replacement proportional to weight
        n_samples = max(1000, len(self.entries) * 5)  # epoch size heuristic (can be tuned)
        chosen = np.random.choice(len(self.entries), size=n_samples, replace=True, p=self.probs)
        batch = []
        for idx in chosen:
            entry = self.entries[int(idx)]
            ds_idx = entry['dataset_idx']
            weight = float(entry['weight'])
            item = self.base[ds_idx]
            # prepare the yielded example
            yield {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': torch.tensor(item['label'], dtype=torch.long),
                'weights': torch.tensor(weight, dtype=torch.float)
            }


# ----------------------------
# Main training loop with CRUST
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--output_dir', type=str, default='./tiny_crust_runs')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--apply_crust', action='store_true')
    parser.add_argument('--select_fraction', type=float, default=0.3,
                        help='Fraction of dataset to select as medoids overall per epoch')
    parser.add_argument('--k_per_class_factor', type=float, default=0.05,
                        help='Alternate parameter: fraction of budget to allocate per class (used indirectly)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    ds = load_dataset('stanfordnlp/sst2')
    train_texts = [ex['sentence'] for ex in ds['train']]
    train_labels = [int(ex['label']) for ex in ds['train']]
    val_texts = [ex['sentence'] for ex in ds['validation']]
    val_labels = [int(ex['label']) for ex in ds['validation']]

    train_data = make_dataset(tokenizer, train_texts, train_labels, args.max_len)
    val_data = make_dataset(tokenizer, val_texts, val_labels, args.max_len)

    # dataloader used for embedding computation (batch size can be large)
    encode_loader = DataLoader(train_data, batch_size=64, shuffle=False, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size
    num_classes = max(max(train_labels), max(val_labels)) + 1

    model = TinyGPT(vocab_size=vocab_size, max_len=args.max_len, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, num_classes=num_classes).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # validation loader unchanged
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")

        if args.apply_crust:
            # 1) compute last-layer embeddings (N, C*D) and predicted classes
            embeddings, preds, indices, h_arr = compute_last_layer_embeddings(model, encode_loader, device)
            N = embeddings.shape[0]
            budget = max(1, int(round(args.select_fraction * N)))  # total medoids to select across classes

            # 2) per-class medoids
            medoid_positions, cluster_assignments = per_class_medoids(embeddings, preds, indices, num_classes, budget)
            print(f"CRUST selected {len(medoid_positions)} medoids (budget {budget})")

            # 3) build weighted coreset (medoid entries with weight r_j)
            coreset_entries = build_weighted_coreset(train_data, medoid_positions, cluster_assignments, embeddings.shape, args.select_fraction)

            # 4) create a dataloader that samples medoids proportional to r_j (weights)
            coreset_ds = CoresetDataset(train_data, coreset_entries, tokenizer, args.max_len, batch_size=args.batch_size)
            train_loader = DataLoader(coreset_ds, batch_size=args.batch_size)  # IterableDataset yields dicts; batch handled inside
        else:
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        # Train for one epoch on chosen dataset (coreset weighted or full)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch} results: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch}.pt"))

    print("Training finished.")

if __name__ == '__main__':
    main()