from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

class AttentionFusionTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl=None
        ):

        super(AttentionFusionTrainer, self).__init__(args)
        self.epoch = 0 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.ehr_model = LSTM(input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)

        # Use our attention-based fusion module
        self.model = AttentionMedFuse(args, self.ehr_model, self.cxr_model).to(self.device)
        
        # Initialize fusion method
        self.init_fusion_method()

        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        self.load_state()
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')
        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {
            'loss train': [], 
            'loss val': [], 
            'auroc val': [], 
            'loss align train': [], 
            'loss align val': [],
            'ehr_attention': [],
            'cxr_attention': []
        }
    
    def init_fusion_method(self):
        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        
        if self.args.load_state is not None:
            self.load_state()

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        epoch_loss_align = 0
        ehr_attention_sum = 0
        cxr_attention_sum = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img, pairs)
            
            pred = output['attention_fusion'].squeeze()
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            
            # Track modality attention weights
            if 'modality_weights' in output:
                ehr_attention_sum += output['modality_weights'][:, 0].mean().item()
                cxr_attention_sum += output['modality_weights'][:, 1].mean().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")
        
        # Store average attention weights per epoch
        self.epochs_stats['ehr_attention'].append(ehr_attention_sum / steps)
        self.epochs_stats['cxr_attention'].append(cxr_attention_sum / steps)
        
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        self.epochs_stats['loss train'].append(epoch_loss/i)
        return ret
    
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        ehr_attention_sum = 0
        cxr_attention_sum = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        
        # Save attention maps for visualization
        all_attention_maps = []
        all_modality_weights = []
        sample_ids = []

        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                
                output = self.model(x, seq_lengths, img, pairs)
                
                pred = output['attention_fusion']
                if len(pred.shape) > 1:
                    pred = pred.squeeze()
                
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                
                # Track modality attention weights
                if 'modality_weights' in output:
                    ehr_attention_sum += output['modality_weights'][:, 0].mean().item()
                    cxr_attention_sum += output['modality_weights'][:, 1].mean().item()
                    
                    # Save examples for visualization
                    all_modality_weights.append(output['modality_weights'].cpu().numpy())
                
                # Save attention maps
                if 'attention_weights' in output:
                    all_attention_maps.append(output['attention_weights'].cpu().numpy())
                    
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
                
                # Store sample IDs for the first few examples
                if len(sample_ids) < 10:
                    batch_ids = list(range(i*dl.batch_size, min((i+1)*dl.batch_size, len(dl.dataset))))
                    sample_ids.extend(batch_ids)
        
        # Save attention visualizations
        if self.epoch % 5 == 0:  # Save every 5 epochs
            self.visualize_attention(all_attention_maps[:10], sample_ids[:10], f"attention_epoch_{self.epoch}")
            self.visualize_modality_weights(all_modality_weights[:10], sample_ids[:10], f"modality_weights_epoch_{self.epoch}")
        
        # Store average attention weights per epoch
        self.epochs_stats['ehr_attention'].append(ehr_attention_sum / len(dl))
        self.epochs_stats['cxr_attention'].append(cxr_attention_sum / len(dl))
            
        self.scheduler.step(epoch_loss/len(dl))

        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
        
        self.epochs_stats['auroc val'].append(ret['auroc_mean'])
        self.epochs_stats['loss val'].append(epoch_loss/i)
        
        return ret
    
    def visualize_attention(self, attention_maps, sample_ids, filename):
        """Visualize attention maps for interpretability"""
        if not attention_maps:
            return
            
        fig, axes = plt.subplots(len(attention_maps), 1, figsize=(10, 2*len(attention_maps)))
        
        if len(attention_maps) == 1:
            axes = [axes]
            
        for i, (attn_map, sample_id) in enumerate(zip(attention_maps, sample_ids)):
            # Normalize to [0, 1] for visualization
            attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # Plot the attention map
            sns.heatmap(attn_map_norm, ax=axes[i], cmap="viridis", cbar=True)
            axes[i].set_title(f"Sample {sample_id}")
            axes[i].set_xlabel("Target Modality")
            axes[i].set_ylabel("Source Modality")
            axes[i].set_xticks([0.5, 1.5])
            axes[i].set_yticks([0.5, 1.5]) 
            axes[i].set_xticklabels(['EHR', 'CXR'])
            axes[i].set_yticklabels(['EHR', 'CXR'])
            
        plt.tight_layout()
        plt.savefig(f"{self.args.save_dir}/{filename}.pdf")
        plt.close()
    
    def visualize_modality_weights(self, modality_weights, sample_ids, filename):
        """Visualize modality importance weights"""
        if not modality_weights:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract weights
        ehr_weights = np.array([w[:, 0] for w in modality_weights]).flatten()
        cxr_weights = np.array([w[:, 1] for w in modality_weights]).flatten()
        
        # Create labels
        labels = ['EHR'] * len(ehr_weights) + ['CXR'] * len(cxr_weights)
        values = np.concatenate([ehr_weights, cxr_weights])
        
        # Create a violin plot of the weights
        sns.violinplot(x=labels, y=values, ax=ax)
        ax.set_title('Modality Importance Weights Distribution')
        ax.set_ylabel('Attention Weight')
        ax.set_ylim(0, 1)
        
        plt.tight_layout() 
        plt.savefig(f"{self.args.save_dir}/{filename}.pdf")
        plt.close()
        
        # Also plot modality weights over training
        if len(self.epochs_stats['ehr_attention']) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(self.epochs_stats['ehr_attention']) + 1)
            ax.plot(epochs, self.epochs_stats['ehr_attention'], label='EHR Attention')
            ax.plot(epochs, self.epochs_stats['cxr_attention'], label='CXR Attention')
            
            ax.set_title('Modality Importance Over Training')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Attention Weight')
            ax.set_ylim(0, 1)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.args.save_dir}/modality_weights_training.pdf")
            plt.close()