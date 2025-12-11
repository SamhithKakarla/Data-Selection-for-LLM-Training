# --------------------------------------------------------------------------------------------------------------------
# This file is based on base_train.py . Jump to line 179-213 for the section of retrieving embeddings
# e.g. python greedy_retrieve_embedding.py --encode_layer 2 --epochs 4
# --------------------------------------------------------------------------------------------------------------------

import os
import argparse
import numpy as np
from tqdm import tqdm
from greedy_model import TinyGPT
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

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
            'label': labels[i]
        })
    return dataset

def compute_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--output_dir', type=str, default='./tiny_gpt_runs')
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_classes', type=int, default=3)
    # Added encode_layer parameter
    parser.add_argument('--encode_layer', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    ds = load_dataset('stanfordnlp/sst2')

    train_texts = [ex['sentence'] for ex in ds['train']]
    train_labels = [ex['label'] for ex in ds['train']]

    val_texts = [ex ['sentence'] for ex in ds['validation']]
    val_labels = [ex['label'] for ex in ds['validation']]


    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Label distribution (train): {set(train_labels)}")

    train_data = make_dataset(tokenizer, train_texts, train_labels, args.max_len)
    val_data = make_dataset(tokenizer, val_texts, val_labels, args.max_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size

    actual_num_classes = max(max(train_labels), max(val_labels)) + 1
    print(f"Number of classes detected: {actual_num_classes}")

    model = TinyGPT(
        vocab_size=vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_classes=actual_num_classes
    ).to(device)

    print('Param count:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch}')

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask) 
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            acc = compute_accuracy(preds, labels)

            total_loss += loss.item()
            total_acc += acc
            steps += 1

            if steps % 50 == 0:
                loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/steps:.4f}',
                    'acc': f'{acc:.4f}',
                    'avg_acc': f'{total_acc/steps:.4f}'
                })

        avg_train_loss = total_loss / max(1, steps)
        avg_train_acc = total_acc / max(1, steps)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        vsteps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

                preds = torch.argmax(logits, dim=-1)
                acc = compute_accuracy(preds, labels)

                val_loss += loss.item()
                val_acc += acc
                vsteps += 1

        avg_val_loss = val_loss / max(1, vsteps)
        avg_val_acc = val_acc / max(1, vsteps)

        print(f'Epoch {epoch} — Train loss: {avg_train_loss:.4f}, Train acc: {avg_train_acc:.4f}, '
              f'Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.4f}')

        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch{epoch}.pt'))

    # -------------------------
    # Extract and save embeddings once per sample
    # -------------------------
    all_embeddings = []
    all_labels = []
    all_sentences = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Extracting embeddings")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()

            sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]

            emb = model.encode(input_ids, attention_mask, args.encode_layer) 

            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels)
            all_sentences.extend(sentences)

    # Stack embeddings and labels
    embeddings_array = np.vstack(all_embeddings) 
    labels_array = np.concatenate(all_labels)   
    sentences_array = np.array(all_sentences, dtype=object) 

    # Save everything together
    np.savez(os.path.join(args.output_dir, "train_embeddings.npz"),
            embeddings=embeddings_array,
            labels=labels_array,
            sentences=sentences_array)

    print(f"Saved embeddings to {os.path.join(args.output_dir, 'train_embeddings.npz')}")

if __name__ == '__main__':
    main()