# train.py
import argparse
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model import TinyGPT


def collate_fn(batch):
    # batch: list of dicts with 'input_ids' tensors and 'label'
    input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, T)
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)  # (B,)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def make_dataset(tokenizer, texts, labels, max_len):
    # Tokenize with padding/truncation to fixed length
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    dataset = []
    for i in range(enc["input_ids"].size(0)):
        dataset.append(
            {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "label": labels[i],
            }
        )
    return dataset


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    loop = tqdm(data_loader, desc=f"Epoch: {epoch}")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        steps += 1

        if steps % 50 == 0 or steps == len(data_loader):
            loop.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/steps:.4f}",
                    "acc": f"{acc:.4f}",
                    "avg_acc": f"{total_acc/steps:.4f}",
                }
            )

    avg_loss = total_loss / max(1, steps)
    avg_acc = total_acc / max(1, steps)
    return avg_loss, avg_acc


def evaluate(model, data_loader, criterion, device, epoch=None, track_correctness=None):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    idx_offset = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean().item()

            total_loss += loss.item()
            total_acc += acc

            # update correctness if tracking
            if track_correctness is not None:
                batch_size = input_ids.size(0)
                track_correctness[idx_offset : idx_offset + batch_size, epoch - 1] = (
                    preds == labels
                ).cpu()
                idx_offset += batch_size

    avg_loss = total_loss / max(1, len(data_loader))
    avg_acc = total_acc / max(1, len(data_loader))
    return avg_loss, avg_acc


def compute_accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./tiny_gpt_runs")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--use_fsa", type=bool, default=False)
    parser.add_argument("--fsa_percent", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    ds = load_dataset("stanfordnlp/sst2")

    train_texts = [ex["sentence"] for ex in ds["train"]]
    train_labels = [ex["label"] for ex in ds["train"]]

    val_texts = [ex["sentence"] for ex in ds["validation"]]
    val_labels = [ex["label"] for ex in ds["validation"]]

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Label distribution (train): {set(train_labels)}")

    train_data = make_dataset(tokenizer, train_texts, train_labels, args.max_len)
    val_data = make_dataset(tokenizer, val_texts, val_labels, args.max_len)

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    vocab_size = tokenizer.vocab_size

    # Determine actual number of classes from data
    actual_num_classes = max(max(train_labels), max(val_labels)) + 1
    print(f"Number of classes detected: {actual_num_classes}")

    model = TinyGPT(
        vocab_size=vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_classes=actual_num_classes,
    ).to(device)

    print("Param count:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Track correctness for forgetting scores only
    num_examples = len(train_data)
    correct_matrix = torch.zeros(num_examples, args.epochs, dtype=torch.bool)

    for epoch in range(1, args.epochs + 1):
        # get train / validation metrics
        avg_train_loss, avg_train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        avg_val_loss, avg_val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            track_correctness=correct_matrix,
        )

        # print metrics and save model
        print(
            f"Epoch {epoch} — Train loss: {avg_train_loss:.4f}, Train acc: {avg_train_acc:.4f}, "
            f"Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.4f}"
        )
        torch.save(
            model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch}.pt")
        )

    # ---- Rerun on the FSA examples ----
    if args.use_fsa:
        print("Running FSA: fine-tuning on hardest examples...")

        forgetting_scores = torch.zeros(num_examples, dtype=torch.int)
        for i in range(num_examples):
            for e in range(args.epochs - 1):
                if correct_matrix[i, e] and not correct_matrix[i, e + 1]:
                    forgetting_scores[i] += 1

        # decide how many examples to keep
        num_keep = int(len(train_data) * args.fsa_percent)
        top_forgetting_indices = torch.argsort(forgetting_scores, descending=True)[
            :num_keep
        ]
        fsa_subset = [train_data[i] for i in top_forgetting_indices]

        fsa_loader = DataLoader(
            fsa_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        fsa_model = TinyGPT(
            vocab_size=vocab_size,
            max_len=args.max_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            num_classes=actual_num_classes,
        ).to(device)
        fsa_optimizer = torch.optim.AdamW(fsa_model.parameters(), lr=args.lr)

        print("Now retraining on only the FSA subset...")
        for epoch in range(1, args.epochs + 1):
            # get train / validation metrics
            avg_train_loss, avg_train_acc = train_one_epoch(
                fsa_model,
                fsa_loader,
                fsa_optimizer,
                criterion,
                device,
                epoch,
            )
            avg_val_loss, avg_val_acc = evaluate(
                fsa_model, val_loader, criterion, device
            )

            # print metrics and save model
            print(
                f"FSA Epoch {epoch} — Train loss: {avg_train_loss:.4f}, Train acc: {avg_train_acc:.4f}, "
                f"Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.4f}"
            )
            torch.save(
                fsa_model.state_dict(),
                os.path.join(args.output_dir, f"fsa_model_epoch{epoch}.pt"),
            )

        print("FSA training complete.")

    print("Done. Models saved to", args.output_dir)


if __name__ == "__main__":
    main()
