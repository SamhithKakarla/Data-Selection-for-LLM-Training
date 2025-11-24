# train.py
import os
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from model import TinyGPT

def collate_fn(batch):
    # batch: list of dicts with 'input_ids' tensors (padded to same length already)
    input_ids = torch.stack([b['input_ids'] for b in batch])  # (B, T)
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

def make_dataset(tokenizer, texts, max_len):
    # Tokenize with padding/truncation to fixed length and return list of tensors
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    dataset = []
    for i in range(enc['input_ids'].size(0)):
        dataset.append({'input_ids': enc['input_ids'][i], 'attention_mask': enc['attention_mask'][i]})
    return dataset

def shift_inputs_targets(batch_input_ids):
    # For causal LM next-token prediction:
    # inputs = tokens[:, :-1], targets = tokens[:, 1:]
    inputs = batch_input_ids[:, :-1].contiguous()
    targets = batch_input_ids[:, 1:].contiguous()
    return inputs, targets

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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    ds = load_dataset('stanfordnlp/sst2')
    train_texts = [ex['sentence'] for ex in ds['train']]
    val_texts   = [ex['sentence'] for ex in ds['validation']]

    # Option: apply data-selection here (coreset). Replace train_texts with subset if desired.

    train_data = make_dataset(tokenizer, train_texts, args.max_len)
    val_data   = make_dataset(tokenizer, val_texts, args.max_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size
    model = TinyGPT(vocab_size=vocab_size, max_len=args.max_len, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads).to(device)

    print('Param count:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in loop:
            input_ids = batch['input_ids'].to(device)            # (B, T)
            attention_mask = batch['attention_mask'].to(device)  # (B, T)

            # Create inputs/targets for causal LM
            inputs, targets = shift_inputs_targets(input_ids)    # inputs: (B, T-1), targets: (B, T-1)

            logits = model(inputs, attention_mask=attention_mask[:, :-1])  # (B, T-1, V)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            loss = criterion(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            if steps % 50 == 0:
                loop.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{total_loss/steps:.4f}'})

        avg_train_loss = total_loss / max(1, steps)

        # validation
        model.eval()
        val_loss = 0.0
        vsteps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                inputs, targets = shift_inputs_targets(input_ids)
                logits = model(inputs, attention_mask=attention_mask[:, :-1])
                l = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += l.item()
                vsteps += 1
        avg_val_loss = val_loss / max(1, vsteps)

        print(f'Epoch {epoch} — Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch{epoch}.pt'))

    print('Done. Models saved to', args.output_dir)

if __name__ == '__main__':
    main()
