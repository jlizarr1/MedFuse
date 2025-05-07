from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from attention_medfuse import TransformerMedFuse
from .trainer import Trainer

class TransformerTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl=None
        ):

        super(TransformerTrainer, self).__init__(args)
        self.epoch = 0 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        # Initialize models
        self.ehr_model = LSTM(input_dim=76, num_classes=args.num_classes, 
                         hidden_dim=args.dim, dropout=args.dropout, 
                         layers=args.layers).to(self.device)
                         
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)

        # Initialize transformer fusion model
        self.model = TransformerMedFuse(args, self.ehr_model, self.cxr_model).to(self.device)
        
        # Initialize cross-modal generation if specified
        if args.cross_modal_generation:
            self.cross_modal_generator = CrossModalGenerator(
                ehr_dim=self.ehr_model.feats_dim,
                cxr_dim=self.cxr_model.feats_dim
            ).to(self.device)
        
        # Load pretrained models if specified
        self.init_fusion_method()

        # Initialize loss, optimizer and scheduler
        self.loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
        
        if args.cross_modal_generation:
            self.generator_optimizer = optim.Adam(
                self.cross_modal_generator.parameters(), 
                args.lr, 
                betas=(0.9, self.args.beta_1)
            )
        
        self.load_state()
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')
        
        # Initialize tracking variables
        self.best_auroc = 0
        self.best_stats = None
        self.epochs_stats = {
            'loss train': [], 
            'loss val': [], 
            'auroc val': [],
            'attention_cls': [],
            'attention_ehr': [],
            'attention_cxr': []
        }
        
        if args.cross_modal_generation:
            self.epochs_stats.update({
                'generator_loss_train': [],
                'generator_loss_val': []
            })
    
    def init_fusion_method(self):
        # Load pretrained models if specified
        if self.args.load_state_ehr is not None:
            self.load_ehr_pheno(load_state=self.args.load_state_ehr)
        if self.args.load_state_cxr is not None:
            self.load_cxr_pheno(load_state=self.args.load_state_cxr)
        
        if self.args.load_state is not None:
            self.load_state()

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        generator_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        steps = len(self.train_dl)
        
        # Track attention weights statistics
        attention_cls_sum = 0
        attention_ehr_sum = 0
        attention_cxr_sum = 0
        
        for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            img = img.to(self.device)

            # Forward pass
            output = self.model(x, seq_lengths, img, pairs)
            
            # Main prediction loss
            pred = output['transformer_fusion'].squeeze()
            loss = self.loss(pred, y)
            epoch_loss += loss.item()
            
            # Train cross-modal generator if enabled
            if self.args.cross_modal_generation and hasattr(self, 'cross_modal_generator'):
                # Get features
                ehr_feats = output['ehr_feats']
                cxr_feats = output['cxr_feats']
                
                # Generate cross-modal features
                generated, gen_losses = self.cross_modal_generator(ehr_feats, cxr_feats)
                
                # Calculate generator loss (MSE + Cosine)
                gen_loss = sum(gen_losses.values())
                generator_loss += gen_loss.item()
                
                # Backpropagate generator loss
                self.generator_optimizer.zero_grad()
                gen_loss.backward(retain_graph=True)
                self.generator_optimizer.step()
            
            # Backpropagate main loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track predictions
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, y), 0)
            
            # Track attention statistics
            if 'attention_weights' in output:
                attn = output['attention_weights']
                # First row: CLS token attending to EHR and CXR
                attention_cls_sum += attn[:, 0, 0].mean().item()  # CLS->CLS
                attention_ehr_sum += attn[:, 0, 1].mean().item()  # CLS->EHR
                attention_cxr_sum += attn[:, 0, 2].mean().item()  # CLS->CXR

            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")
                if self.args.cross_modal_generation:
                    print(f" Generator loss: {generator_loss/i:0.5f}")
        
        # Store attention statistics
        self.epochs_stats['attention_cls'].append(attention_cls_sum / steps)
        self.epochs_stats['attention_ehr'].append(attention_ehr_sum / steps)
        self.epochs_stats['attention_cxr'].append(attention_cxr_sum / steps)
        
        # Store losses
        self.epochs_stats['loss train'].append(epoch_loss/i)
        if self.args.cross_modal_generation:
            self.epochs_stats['generator_loss_train'].append(generator_loss/i)
        
        # Calculate performance metrics
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'train')
        return ret
    
    def validate(self, dl):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        generator_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        
        # Track attention weights statistics
        attention_cls_sum = 0
        attention_ehr_sum = 0
        attention_cxr_sum = 0
        
        # Save attention maps for visualization
        all_attention_maps = []
        sample_ids = []
        
        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, pairs) in enumerate(dl):
                y = self.get_gt(y_ehr, y_cxr)
                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)
                
                # Forward pass
                output = self.model(x, seq_lengths, img, pairs)
                
                # Main prediction loss
                pred = output['transformer_fusion']
                if len(pred.shape) > 1:
                    pred = pred.squeeze()
                
                loss = self.loss(pred, y)
                epoch_loss += loss.item()
                
                # Calculate generator loss if enabled
                if self.args.cross_modal_generation and hasattr(self, 'cross_modal_generator'):
                    # Get features
                    ehr_feats = output['ehr_feats']
                    cxr_feats = output['cxr_feats']
                    
                    # Generate cross-modal features
                    generated, gen_losses = self.cross_modal_generator(ehr_feats, cxr_feats)
                    
                    # Calculate generator loss (MSE + Cosine)
                    gen_loss = sum(gen_losses.values())
                    generator_loss += gen_loss.item()
                
                # Track predictions
                outPRED = torch.cat((outPRED, pred), 0)
                outGT = torch.cat((outGT, y), 0)
                
                # Track attention statistics
                if 'attention_weights' in output:
                    attn = output['attention_weights']
                    # First row: CLS token attending to EHR and CXR
                    attention_cls_sum += attn[:, 0, 0].mean().item()  # CLS->CLS
                    attention_ehr_sum += attn[:, 0, 1].mean().item()  # CLS->EHR
                    attention_cxr_sum += attn[:, 0, 2].mean().item()  # CLS->CXR
                    
                    # Save attention maps for visualization
                    if self.args.visualize_attention and len(all_attention_maps) < 10:
                        all_attention_maps.append(attn[0].cpu().numpy())  # Save first sample from batch
                        sample_ids.append(i * dl.batch_size)
        
        # Visualize attention maps if enabled
        if self.args.visualize_attention and self.epoch % 5 == 0 and all_attention_maps:
            self.visualize_transformer_attention(all_attention_maps, sample_ids)
        
        # Store attention statistics
        self.epochs_stats['attention_cls'].append(attention_cls_sum / len(dl))
        self.epochs_stats['attention_ehr'].append(attention_ehr_sum / len(dl))
        self.epochs_stats['attention_cxr'].append(attention_cxr_sum / len(dl))
        
        # Update learning rate
        self.scheduler.step(epoch_loss/len(dl))
        
        # Store losses
        self.epochs_stats['loss val'].append(epoch_loss/i)
        if self.args.cross_modal_generation:
            self.epochs_stats['generator_loss_val'].append(generator_loss/i)
        
        # Calculate and log metrics
        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
        self.epochs_stats['auroc val'].append(ret['auroc_mean'])
        
        return ret
    
    def visualize_transformer_attention(self, attention_maps, sample_ids):
        """Visualize transformer attention maps for interpretability"""
        if not attention_maps:
            return
            
        # Create directory for visualizations
        viz_dir = os.path.join(self.args.save_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot each attention map
        for i, (attn_map, sample_id) in enumerate(zip(attention_maps, sample_ids)):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Normalize attention weights for better visualization
            attn_map_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # Create heatmap
            sns.heatmap(attn_map_norm, ax=ax, cmap="viridis", cbar=True, 
                       xticklabels=['CLS', 'EHR', 'CXR'], 
                       yticklabels=['CLS', 'EHR', 'CXR'])
            
            ax.set_title(f"Sample {sample_id} - Transformer Attention Map")
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Token Position")
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"attention_map_sample_{sample_id}_epoch_{self.epoch}.pdf"))
            plt.close()
        
        # Also visualize attention statistics over training
        if len(self.epochs_stats['attention_ehr']) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(self.epochs_stats['attention_ehr']) + 1)
            ax.plot(epochs, self.epochs_stats['attention_cls'], label='CLS Token Attention')
            ax.plot(epochs, self.epochs_stats['attention_ehr'], label='EHR Attention')
            ax.plot(epochs, self.epochs_stats['attention_cxr'], label='CXR Attention')
            
            ax.set_title('Attention Distribution Over Training')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Attention Weight')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"attention_distribution_epoch_{self.epoch}.pdf"))
            plt.close()