#!/usr/bin/env python
"""
Analysis script to understand why certain uni-modal ratios perform better.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from arguments import args_parser
from ehr_utils.preprocessing import Discretizer, Normalizer
from datasets.ehr_dataset import get_datasets
from datasets.cxr_dataset import get_cxr_datasets
from datasets.fusion import load_cxr_ehr
from trainers.fusion_trainer import FusionTrainer

def extract_features(model, data_loader, device):
    """Extract features from the model for analysis."""
    model.eval()
    ehr_features = []
    cxr_features = []
    fused_features = []
    labels = []
    is_paired = []
    
    with torch.no_grad():
        for x, img, y_ehr, y_cxr, seq_lengths, pairs in data_loader:
            x = torch.from_numpy(x).float().to(device)
            img = img.to(device)
            y = torch.from_numpy(y_ehr).float().to(device)
            
            # Forward pass
            output = model(x, seq_lengths, img, pairs)
            
            # Extract features
            if 'ehr_feats' in output:
                ehr_features.append(output['ehr_feats'].cpu().numpy())
            
            if 'cxr_feats' in output:
                cxr_features.append(output['cxr_feats'].cpu().numpy())
            
            # For LSTM fusion, get the last hidden state
            if 'lstm' in model.args.fusion_type:
                hidden_state = model.model.lstm_fusion_layer[0].h.cpu().numpy()
                fused_features.append(hidden_state)
            
            # Store labels and pairing info
            labels.append(y.cpu().numpy())
            is_paired.append(np.array(pairs))
    
    # Concatenate results
    ehr_features = np.concatenate(ehr_features, axis=0) if ehr_features else None
    cxr_features = np.concatenate(cxr_features, axis=0) if cxr_features else None
    fused_features = np.concatenate(fused_features, axis=0) if fused_features else None
    labels = np.concatenate(labels, axis=0)
    is_paired = np.concatenate(is_paired, axis=0)
    
    return {
        'ehr_features': ehr_features,
        'cxr_features': cxr_features,
        'fused_features': fused_features,
        'labels': labels,
        'is_paired': is_paired
    }

def visualize_features(features, labels, is_paired, title, save_path):
    """Visualize feature embeddings using t-SNE."""
    # Perform dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    pca = PCA(n_components=50, random_state=42)
    
    # Apply PCA first for efficiency if features are high-dimensional
    if features.shape[1] > 50:
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features
    
    # Apply t-SNE
    embedding = tsne.fit_transform(features_reduced)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot colored by paired status
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=is_paired.astype(int), 
        cmap='coolwarm', 
        alpha=0.7,
        s=50
    )
    
    # Add legend
    plt.colorbar(scatter, label='Is Paired')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    return embedding

def analyze_feature_distances(features, is_paired, title, save_path):
    """Analyze the distances between features based on pairing status."""
    # Split features into paired and unpaired
    paired_features = features[is_paired == True]
    unpaired_features = features[is_paired == False]
    
    # Sample equal number of features from both sets if needed
    min_samples = min(len(paired_features), len(unpaired_features))
    if min_samples < len(paired_features):
        paired_idx = np.random.choice(len(paired_features), min_samples, replace=False)
        paired_features = paired_features[paired_idx]
    
    if min_samples < len(unpaired_features):
        unpaired_idx = np.random.choice(len(unpaired_features), min_samples, replace=False)
        unpaired_features = unpaired_features[unpaired_idx]
    
    # Calculate distances
    from scipy.spatial.distance import pdist, squareform
    
    # Within paired samples
    paired_distances = pdist(paired_features)
    
    # Within unpaired samples
    unpaired_distances = pdist(unpaired_features)
    
    # Between paired and unpaired samples
    paired_unpaired_distances = []
    for p_feat in paired_features:
        for u_feat in unpaired_features:
            paired_unpaired_distances.append(np.linalg.norm(p_feat - u_feat))
    
    paired_unpaired_distances = np.array(paired_unpaired_distances)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    plt.hist([paired_distances, unpaired_distances, paired_unpaired_distances], 
             bins=50, 
             alpha=0.7, 
             label=['Within Paired', 'Within Unpaired', 'Between Paired-Unpaired'])
    
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(f'Feature Distance Distributions - {title}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    
    # Calculate statistics
    stats = {
        'mean_paired_dist': np.mean(paired_distances),
        'std_paired_dist': np.std(paired_distances),
        'mean_unpaired_dist': np.mean(unpaired_distances),
        'std_unpaired_dist': np.std(unpaired_distances),
        'mean_between_dist': np.mean(paired_unpaired_distances),
        'std_between_dist': np.std(paired_unpaired_distances),
    }
    
    return stats

def analyze_models_across_ratios(args):
    """Analyze models trained with different uni-modal ratios."""
    # Set up directories
    analysis_dir = os.path.join(args.save_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up data preprocessing
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
    
    # Load datasets
    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)
    cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(args)
    
    # Analyze across different ratios
    ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    models = ['lstm']  # Focus on MedFuse
    
    results = []
    
    for model_type in models:
        for ratio in ratios:
            args.fusion_type = model_type
            args.data_ratio = ratio
            
            # Create data loaders
            train_dl, val_dl, test_dl = load_cxr_ehr(args, ehr_train_ds, ehr_val_ds, cxr_train_ds, 
                                                   cxr_val_ds, ehr_test_ds, cxr_test_ds)
            
            # Initialize trainer
            trainer = FusionTrainer(train_dl, val_dl, args, test_dl=test_dl)
            
            # Load model if trained
            model_path = os.path.join(args.save_dir, f'{model_type}_ratio_{ratio}', 'best_checkpoint.pth.tar')
            if os.path.exists(model_path):
                args.load_state = model_path
                trainer.load_state()
                print(f"Loaded model: {model_path}")
            else:
                print(f"Model not found: {model_path}, using untrained model")
            
            # Extract features
            features = extract_features(trainer.model, val_dl, device)
            
            # Visualize features
            if features['ehr_features'] is not None:
                embedding = visualize_features(
                    features['ehr_features'], 
                    features['labels'], 
                    features['is_paired'],
                    f"EHR Features - {model_type} - Ratio {ratio}",
                    os.path.join(analysis_dir, f"ehr_tsne_{model_type}_ratio_{ratio}.pdf")
                )
            
            if features['cxr_features'] is not None:
                embedding = visualize_features(
                    features['cxr_features'], 
                    features['labels'], 
                    features['is_paired'],
                    f"CXR Features - {model_type} - Ratio {ratio}",
                    os.path.join(analysis_dir, f"cxr_tsne_{model_type}_ratio_{ratio}.pdf")
                )
            
            if features['fused_features'] is not None:
                embedding = visualize_features(
                    features['fused_features'], 
                    features['labels'], 
                    features['is_paired'],
                    f"Fused Features - {model_type} - Ratio {ratio}",
                    os.path.join(analysis_dir, f"fused_tsne_{model_type}_ratio_{ratio}.pdf")
                )
            
            # Analyze feature distances
            if features['ehr_features'] is not None:
                stats_ehr = analyze_feature_distances(
                    features['ehr_features'],
                    features['is_paired'],
                    f"EHR Features - {model_type} - Ratio {ratio}",
                    os.path.join(analysis_dir, f"ehr_distances_{model_type}_ratio_{ratio}.pdf")
                )
                
                stats_ehr.update({
                    'model_type': model_type,
                    'ratio': ratio,
                    'feature_type': 'ehr'
                })
                
                results.append(stats_ehr)
            
            if features['cxr_features'] is not None:
                stats_cxr = analyze_feature_distances(
                    features['cxr_features'],
                    features['is_paired'],
                    f"CXR Features - {model_type} - Ratio {ratio}",
                    os.path.join(analysis_dir, f"cxr_distances_{model_type}_ratio_{ratio}.pdf")
                )
                
                stats_cxr.update({
                    'model_type': model_type,
                    'ratio': ratio,
                    'feature_type': 'cxr'
                })
                
                results.append(stats_cxr)
            
            if features['fused_features'] is not None:
                stats_fused = analyze_feature_distances(
                    features['fused_features'],
                    features['is_paired'],
                    f"Fused Features - {model_type} - Ratio {ratio}",
                    os.path.join(analysis_dir, f"fused_distances_{model_type}_ratio_{ratio}.pdf")
                )
                
                stats_fused.update({
                    'model_type': model_type,
                    'ratio': ratio,
                    'feature_type': 'fused'
                })
                
                results.append(stats_fused)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(analysis_dir, 'feature_distance_analysis.csv'), index=False)
    
    # Plot results
    plot_distance_results(results_df, analysis_dir)
    
    return results_df

def plot_distance_results(df, save_dir):
    """Plot analysis results."""
    # Plot mean distances across ratios
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each feature type
    feature_types = df['feature_type'].unique()
    fig, axes = plt.subplots(len(feature_types), 1, figsize=(12, 5*len(feature_types)))
    
    if len(feature_types) == 1:
        axes = [axes]
    
    for i, feature_type in enumerate(feature_types):
        # Filter data for this feature type
        feature_df = df[df['feature_type'] == feature_type]
        
        # Plot distances across ratios
        sns.lineplot(
            data=feature_df, 
            x='ratio', 
            y='mean_paired_dist', 
            marker='o', 
            label='Within Paired',
            ax=axes[i]
        )
        
        sns.lineplot(
            data=feature_df, 
            x='ratio', 
            y='mean_unpaired_dist', 
            marker='s', 
            label='Within Unpaired',
            ax=axes[i]
        )
        
        sns.lineplot(
            data=feature_df, 
            x='ratio', 
            y='mean_between_dist', 
            marker='^', 
            label='Between Paired-Unpaired',
            ax=axes[i]
        )
        
        axes[i].set_title(f'{feature_type.upper()} Features - Mean Distances Across Ratios')
        axes[i].set_xlabel('Uni-modal Sample Ratio')
        axes[i].set_ylabel('Mean Distance')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_distances_across_ratios.pdf'))
    plt.close()
    
    # Plot ratio vs distance differences
    plt.figure(figsize=(12, 8))
    
    for feature_type in feature_types:
        feature_df = df[df['feature_type'] == feature_type]
        
        # Calculate difference between within-class and between-class distances
        feature_df['within_between_diff'] = feature_df['mean_between_dist'] - (
            (feature_df['mean_paired_dist'] + feature_df['mean_unpaired_dist']) / 2
        )
        
        # Plot difference vs ratio
        sns.lineplot(
            data=feature_df,
            x='ratio',
            y='within_between_diff',
            marker='o',
            label=f'{feature_type.upper()} Features'
        )
    
    plt.title('Distance Difference (Between - Within) Across Ratios')
    plt.xlabel('Uni-modal Sample Ratio')
    plt.ylabel('Distance Difference')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'distance_difference_across_ratios.pdf'))
    plt.close()

def main():
    parser = args_parser()
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Analyze models across different uni-modal ratios
    results = analyze_models_across_ratios(args)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()