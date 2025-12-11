# experiment_crust_vs_random.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

# Import all functions from train_crust
from train_with_crust import (
    TinyGPT, collate_fn, make_dataset,
    compute_last_layer_embeddings, per_class_medoids,
    build_weighted_coreset, CoresetDataset,
    train_one_epoch, evaluate
)


def train_with_random_sampling(model, train_data, val_loader, optimizer, device, 
                               num_epochs, sample_fraction, batch_size):
    
    n_samples = max(1, int(round(len(train_data) * sample_fraction)))
    random_indices = np.random.choice(len(train_data), size=n_samples, replace=False)
    
    print(f"  Random sampling: selected {n_samples} samples")
    
    # Create subset dataset
    subset_data = [train_data[i] for i in random_indices]
    train_loader = DataLoader(subset_data, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    
    val_accs = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    return val_accs


def train_with_crust_sampling(model, train_data, encode_loader, val_loader, optimizer, 
                              device, num_classes, tokenizer, num_epochs, sample_fraction, batch_size):
    """Train with CRUST coreset selection (updates every epoch)."""
    val_accs = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"  Epoch {epoch}: Computing CRUST coreset...")
        
        # Compute embeddings and select coresets (same as original)
        embeddings, preds, indices, h_arr = compute_last_layer_embeddings(
            model, encode_loader, device
        )
        N = embeddings.shape[0]
        budget = max(1, int(round(sample_fraction * N)))
        
        medoid_positions, cluster_assignments = per_class_medoids(
            embeddings, preds, indices, num_classes, budget
        )
        print(f"  CRUST selected {len(medoid_positions)} medoids (budget {budget})")
        
        coreset_entries = build_weighted_coreset(
            train_data, medoid_positions, cluster_assignments, 
            embeddings.shape, sample_fraction
        )
        
        coreset_ds = CoresetDataset(train_data, coreset_entries, tokenizer, 64, batch_size)
        train_loader = DataLoader(coreset_ds, batch_size=batch_size)
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    return val_accs


def run_experiment():
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data (same as original)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    ds = load_dataset('stanfordnlp/sst2')
    train_texts = [ex['sentence'] for ex in ds['train']]
    train_labels = [int(ex['label']) for ex in ds['train']]
    val_texts = [ex['sentence'] for ex in ds['validation']]
    val_labels = [int(ex['label']) for ex in ds['validation']]
    
    train_data = make_dataset(tokenizer, train_texts, train_labels, max_len=64)
    val_data = make_dataset(tokenizer, val_texts, val_labels, max_len=64)
    
    encode_loader = DataLoader(train_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    vocab_size = tokenizer.vocab_size
    num_classes =  max(max(train_labels), max(val_labels)) + 1
    num_epochs = 15
    batch_size = 32
    
    # Experiment parameters
    fractions = [0.4, 0.5]
    results = {
        'fractions': fractions,
        'crust': {},
        'random': {}
    }
    
    # Run experiments
    for frac in fractions:
        print(f"\n{'='*70}")
        print(f"Running with {frac*100:.0f}% of data (~{int(67349*frac):,} samples)")
        print(f"{'='*70}")
        
        # CRUST
        print(f"\n--- CRUST {frac*100:.0f}% ---")
        model_crust = TinyGPT(
            vocab_size=vocab_size, max_len=64, d_model=128, 
            n_layers=4, n_heads=4, num_classes=num_classes
        ).to(device)
        optimizer_crust = torch.optim.AdamW(model_crust.parameters(), lr=3e-4, weight_decay=0.01)
        
        crust_val_accs = train_with_crust_sampling(
            model_crust, train_data, encode_loader, val_loader, 
            optimizer_crust, device, num_classes, tokenizer, num_epochs, frac, batch_size
        )
        results['crust'][frac] = crust_val_accs
        
        # Random sampling
        print(f"\n--- Random {frac*100:.0f}% ---")
        model_random = TinyGPT(
            vocab_size=vocab_size, max_len=64, d_model=128,
            n_layers=4, n_heads=4, num_classes=num_classes
        ).to(device)
        optimizer_random = torch.optim.AdamW(model_random.parameters(), lr=3e-4, weight_decay=0.01)
        
        random_val_accs = train_with_random_sampling(
            model_random, train_data, val_loader, optimizer_random, 
            device, num_epochs, frac, batch_size
        )
        results['random'][frac] = random_val_accs
        
        # Cleanup
        del model_crust, model_random, optimizer_crust, optimizer_random
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    os.makedirs('./experiment_results_2', exist_ok=True)
    with open('./experiment_results_2/crust_vs_random.json', 'w') as f:
        # Convert float keys to strings for JSON
        json_results = {
            'fractions': [float(f) for f in fractions],
            'crust': {str(k): v for k, v in results['crust'].items()},
            'random': {str(k): v for k, v in results['random'].items()}
        }
        json.dump(json_results, f, indent=2)
    
    print("\nResults saved to ./experiment_results_2/crust_vs_random.json")
    
    # Plot results
    plot_results(results)
    
    return results


def plot_results(results):
    """Create comparison plots."""
    fractions = results['fractions']
    epochs = range(1, 16)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CRUST vs Random Sampling: Validation Accuracy Over Training', 
                 fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, frac in enumerate(fractions):
        ax = axes[idx]
        
        crust_accs = results['crust'][frac]
        random_accs = results['random'][frac]
        
        ax.plot(epochs, crust_accs, 'b-o', label='CRUST', linewidth=2.5, markersize=5)
        ax.plot(epochs, random_accs, 'r--s', label='Random', linewidth=2.5, markersize=5)
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'{frac*100:.0f}% of Data ({int(67349*frac):,} samples)', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.50, 0.85])
        
        # Add final accuracy values
        final_crust = crust_accs[-1]
        final_random = random_accs[-1]
        improvement = ((final_crust - final_random) / final_random) * 100
        
        ax.text(0.05, 0.95, f'CRUST: {final_crust:.1%}\nRandom: {final_random:.1%}\n({improvement:+.1f}%)', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('./experiment_results_2/crust_vs_random_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to ./experiment_results_2/crust_vs_random_plot.png")
    
    # Create summary plot: final accuracy vs data fraction
    fig, ax = plt.subplots(figsize=(12, 7))
    
    crust_final = [results['crust'][f][-1] for f in fractions]
    random_final = [results['random'][f][-1] for f in fractions]
    
    x_labels = [f"{int(f*100)}%" for f in fractions]
    x = np.arange(len(fractions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, crust_final, width, label='CRUST', 
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, random_final, width, label='Random', 
                   color='salmon', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Data Fraction', fontsize=13, fontweight='bold')
    ax.set_ylabel('Final Validation Accuracy (Epoch 15)', fontsize=13, fontweight='bold')
    ax.set_title('CRUST vs Random Sampling: Final Performance Comparison', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0.50, 0.85])
    
    # Add improvement percentages on bars
    for i, (c, r) in enumerate(zip(crust_final, random_final)):
        improvement = ((c - r) / r) * 100
        height = max(c, r)
        ax.text(i, height + 0.01, f'+{improvement:.1f}%', 
               ha='center', fontsize=10, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('./experiment_results_2/crust_vs_random_summary.png', dpi=300, bbox_inches='tight')
    print("Summary plot saved to ./experiment_results_2/crust_vs_random_summary.png\n")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("FINAL ACCURACY COMPARISON (Epoch 15)")
    print("="*70)
    print(f"{'Fraction':<12} {'CRUST':<12} {'Random':<12} {'Improvement':<12}")
    print("-"*70)
    for frac, c, r in zip(fractions, crust_final, random_final):
        imp = ((c - r) / r) * 100
        print(f"{frac*100:>6.0f}%     {c:>7.2%}      {r:>7.2%}      {imp:>+7.2f}%")
    print("="*70)
    
    avg_improvement = np.mean([((c - r) / r) * 100 for c, r in zip(crust_final, random_final)])
    print(f"\nAverage Improvement: {avg_improvement:+.2f}%")


if __name__ == '__main__':
    import sys
    print("Starting CRUST vs Random Sampling Experiment")
    print("This will take approximately 2 hours on CPU...\n")
    
    results = run_experiment()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("Results saved in ./experiment_results_2/")
    print("  - crust_vs_random.json (raw data)")
    print("  - crust_vs_random_plot.png (learning curves)")
    print("  - crust_vs_random_summary.png (final comparison)")