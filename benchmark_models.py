#!/usr/bin/env python
"""
Benchmark script to compare different MedFuse models and extensions.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from arguments import args_parser
from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr

def run_benchmark(args):
    """Run benchmark on all fusion methods and collect results."""
    results = []
    fusion_methods = [
        'lstm',                # Original MedFuse
        'attention',           # Attention-based MedFuse
        'transformer',         # Transformer-based MedFuse
        'early',               # Early fusion baseline
        'joint',               # Joint fusion baseline
        'mmtm',                # MMTM baseline
        'daft'                 # DAFT baseline
    ]
    
    # Different uni-modal ratios to test
    uni_modal_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    # Whether to use cross-modal generation
    use_generation = [False, True]
    
    # Set up data preprocessing
    discretizer, normalizer = setup_preprocessing(args)
    
    # Load datasets
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)
    
    # Run benchmarks
    for fusion_method in fusion_methods:
        for ratio in uni_modal_ratios:
            for use_gen in use_generation:
                # Skip cross-modal generation for baseline methods
                if use_gen and fusion_method not in ['lstm', 'attention', 'transformer']:
                    continue
                
                # Set args for this run
                args.fusion_module = fusion_method
                args.data_ratio = ratio
                args.cross_modal_generation = use_gen
                
                # Create data loaders
                train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, 
                                                        cxr_val_ds, ehr_test_ds, cxr_test_ds)
                
                # Run evaluation
                result = evaluate_model(args, train_dl, val_dl, test_dl)
                
                # Add metadata
                result['fusion_method'] = fusion_method
                result['uni_modal_ratio'] = ratio
                result['cross_modal_generation'] = use_gen
                
                # Add to results
                results.append(result)
                
                # Save interim results
                save_results(results, args.save_dir)
    
    return results

def setup_preprocessing(args):
    """Set up data preprocessing utilities."""
    discretizer = Discretizer(timestep=float(args.timestep),
                             store_masks=True,
                             impute_strategy='previous',
                             start_time='zero')
    
    # Get sample data for discretizer
    path = f'{args.ehr_data_dir}/{args.task}/train/14991576_episode3_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    sample_data = np.stack(ret)
    
    # Set up discretizer and normalizer
    discretizer_header = discretizer.transform(sample_data)[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]
    
    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = f'normalizers/ph_ts{args.timestep}.input_str:previous.start_time:zero.normalizer'
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    
    return discretizer, normalizer

def evaluate_model(args, train_dl, val_dl, test_dl):
    """Evaluate a model with the given configuration."""
    # Choose appropriate trainer based on fusion method
    if args.fusion_module == 'attention':
        trainer = AttentionFusionTrainer(train_dl, val_dl, args, test_dl=test_dl)
    elif args.fusion_module == 'transformer':
        trainer = TransformerTrainer(train_dl, val_dl, args, test_dl=test_dl)
    elif args.fusion_module == 'mmtm':
        trainer = MMTMTrainer(train_dl, val_dl, args, test_dl=test_dl)
    elif args.fusion_module == 'daft':
        trainer = DAFTTrainer(train_dl, val_dl, args, test_dl=test_dl)
    else:  # Default MedFuse
        trainer = FusionTrainer(train_dl, val_dl, args, test_dl=test_dl)
    
    # Load pretrained model if available
    if args.load_state is not None:
        trainer.load_state()
    
    # Evaluate model
    print(f"Evaluating {args.fusion_module} with uni_modal_ratio={args.data_ratio}, cross_modal_generation={args.cross_modal_generation}")
    trainer.epoch = 0
    trainer.model.eval()
    
    # Evaluate on validation set
    val_results = trainer.validate(val_dl)
    
    # Evaluate on test set
    test_results = trainer.validate(test_dl)
    
    # Compile results
    results = {
        'val_auroc': val_results['auroc_mean'],
        'val_auprc': val_results['auprc_mean'],
        'test_auroc': test_results['auroc_mean'],
        'test_auprc': test_results['auprc_mean'],
    }
    
    return results

def save_results(results, save_dir):
    """Save benchmark results to CSV and generate plots."""
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(os.path.join(save_dir, 'benchmark_results.csv'), index=False)
    
    # Generate plots
    plot_results(df, save_dir)
    
def plot_results(df, save_dir):
    """Generate plots from benchmark results."""
    # Create visualizations directory
    viz_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot AUROC vs uni-modal ratio by fusion method
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='uni_modal_ratio', y='test_auroc', hue='fusion_method', style='cross_modal_generation')
    plt.title('Test AUROC vs. Uni-modal Ratio')
    plt.xlabel('Uni-modal Sample Ratio')
    plt.ylabel('AUROC')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'test_auroc_vs_ratio.pdf'))
    plt.close()
    
    # Plot AUPRC vs uni-modal ratio by fusion method
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='uni_modal_ratio', y='test_auprc', hue='fusion_method', style='cross_modal_generation')
    plt.title('Test AUPRC vs. Uni-modal Ratio')
    plt.xlabel('Uni-modal Sample Ratio')
    plt.ylabel('AUPRC')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, 'test_auprc_vs_ratio.pdf'))
    plt.close()
    
    # Compare fusion methods (bar plot)
    plt.figure(figsize=(14, 8))
    
    # Get best result for each fusion method
    best_results = df.loc[df.groupby('fusion_method')['test_auroc'].idxmax()]
    
    # Plot bar chart
    sns.barplot(data=best_results, x='fusion_method', y='test_auroc')
    plt.title('Best Test AUROC by Fusion Method')
    plt.xlabel('Fusion Method')
    plt.ylabel('AUROC')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    # Add text labels
    for i, row in enumerate(best_results.itertuples()):
        plt.text(i, row.test_auroc + 0.01, f'{row.test_auroc:.3f}', ha='center')
        plt.text(i, row.test_auroc - 0.05, f'ratio: {row.uni_modal_ratio}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'best_auroc_by_method.pdf'))
    plt.close()

def main():
    # Parse arguments
    parser = args_parser()
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run benchmark
    results = run_benchmark(args)
    
    # Save final results
    save_results(results, args.save_dir)
    
    print("Benchmark completed successfully!")

if __name__ == "__main__":
    main()