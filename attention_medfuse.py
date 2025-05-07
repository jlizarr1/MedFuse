import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class AttentionMedFuse(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
        super(AttentionMedFuse, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = self.args.num_classes
        lstm_in = self.ehr_model.feats_dim
        lstm_out = self.cxr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            lstm_in = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim

        # Projection layer to match dimensions
        self.projection = nn.Linear(projection_in, lstm_in)
        
        # Multi-head attention layer to capture importance of modalities
        self.attention = nn.MultiheadAttention(embed_dim=lstm_in, num_heads=4, batch_first=True)
        
        # LSTM fusion module with attention
        self.lstm_fusion_layer = nn.LSTM(
            lstm_in, lstm_out,
            batch_first=True,
            dropout=0.0)
            
        # Modality importance weighting
        self.modality_attn = nn.Sequential(
            nn.Linear(lstm_in*2, 2),
            nn.Softmax(dim=1)
        )
        
        # Final classification layer
        self.lstm_fused_cls = nn.Sequential(
            nn.Linear(lstm_out, target_classes),
            nn.Sigmoid()
        )

    def forward(self, x, seq_lengths=None, img=None, pairs=None):
        # Get feature representations from modality-specific encoders
        _, ehr_feats = self.ehr_model(x, seq_lengths)
        _, _, cxr_feats = self.cxr_model(img)
        cxr_feats = self.projection(cxr_feats)

        # Create modality sequence and apply attention
        if len(ehr_feats.shape) == 1:
            ehr_feats = ehr_feats.unsqueeze(0)
        
        # Handle missing CXR modality using mask for attention
        cxr_feats_masked = cxr_feats.clone()
        if pairs is not None:
            cxr_feats_masked[list(~np.array(pairs))] = 0

        # Sequence of modalities [ehr, cxr]
        feats = torch.stack([ehr_feats, cxr_feats_masked], dim=1)
        
        # Calculate modality importance weights
        modality_weights = self.modality_attn(torch.cat([ehr_feats, cxr_feats_masked], dim=1))
        
        # Apply multi-head attention to learn cross-modal relationships
        attn_output, attn_weights = self.attention(feats, feats, feats)
        
        # Pack sequences for LSTM and account for missing modalities
        seq_lengths_tensor = torch.ones(len(seq_lengths), dtype=torch.long)
        seq_lengths_tensor[pairs] = 2  # Both modalities present
        
        packed_feats = torch.nn.utils.rnn.pack_padded_sequence(
            attn_output, seq_lengths_tensor.cpu(), 
            batch_first=True, enforce_sorted=False
        )

        # Process with LSTM fusion layer
        _, (hidden, _) = self.lstm_fusion_layer(packed_feats)
        
        # Final classification
        out = hidden.squeeze()
        preds = self.lstm_fused_cls(out)

        return {
            'attention_fusion': preds,
            'ehr_feats': ehr_feats,
            'cxr_feats': cxr_feats_masked,
            'modality_weights': modality_weights,
            'attention_weights': attn_weights
        }

class TransformerMedFuse(nn.Module):
    def __init__(self, args, ehr_model, cxr_model):
        super(TransformerMedFuse, self).__init__()
        self.args = args
        self.ehr_model = ehr_model
        self.cxr_model = cxr_model

        target_classes = self.args.num_classes
        embed_dim = self.ehr_model.feats_dim
        projection_in = self.cxr_model.feats_dim

        if self.args.labels_set == 'radiology':
            target_classes = self.args.vision_num_classes
            embed_dim = self.cxr_model.feats_dim
            projection_in = self.ehr_model.feats_dim

        # Projection layer
        self.projection = nn.Linear(projection_in, embed_dim)
        
        # Position encoding for sequence
        self.position_embedding = nn.Parameter(torch.zeros(1, 3, embed_dim))
        
        # Modality type embeddings (EHR, CXR, CLS)
        self.modality_embeddings = nn.Embedding(3, embed_dim)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=2
        )
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, target_classes),
            nn.Sigmoid()
        )
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize position embeddings
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize projection and classifier
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.zeros_(self.classifier[0].bias)
    
    def forward(self, x, seq_lengths=None, img=None, pairs=None):
        batch_size = x.shape[0]
        
        # Extract features from each modality
        _, ehr_feats = self.ehr_model(x, seq_lengths)
        _, _, cxr_feats = self.cxr_model(img)
        cxr_feats = self.projection(cxr_feats)
        
        # Create CLS token for each item in batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Create sequence with CLS + EHR + CXR
        if len(ehr_feats.shape) == 1:
            ehr_feats = ehr_feats.unsqueeze(0)
            
        ehr_feats = ehr_feats.unsqueeze(1)  # Add sequence dimension
        cxr_feats = cxr_feats.unsqueeze(1)  # Add sequence dimension
        
        # Mask out CXR features for unpaired samples
        if pairs is not None:
            cxr_feats[list(~np.array(pairs))] = 0
            
        # Concatenate tokens into sequence [CLS, EHR, CXR]
        x = torch.cat([cls_tokens, ehr_feats, cxr_feats], dim=1)
        
        # Add positional encoding
        x = x + self.position_embedding
        
        # Add modality type embeddings
        modality_ids = torch.tensor([0, 1, 2]).expand(batch_size, 3).to(x.device)
        x = x + self.modality_embeddings(modality_ids)
        
        # Create attention mask to handle unpaired samples
        mask = None
        if pairs is not None:
            # Create mask where paired samples can attend to all positions
            # Unpaired samples don't attend to CXR position
            mask = torch.ones(batch_size, 3, 3).to(x.device)
            mask[~np.array(pairs), :, 2] = 0  # Don't attend to CXR for unpaired samples
            mask = mask.bool()
            
        # Process sequence through transformer
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use CLS token output for classification
        cls_output = transformer_output[:, 0]
        
        # Classification
        logits = self.classifier(cls_output)
        
        # Calculate attention visualization (last layer only)
        with torch.no_grad():
            # Get attention weights from last transformer layer
            attn_weights = self.transformer_encoder.layers[-1].self_attn.get_attn_output_weights.mean(dim=1)
        
        return {
            'transformer_fusion': logits,
            'ehr_feats': ehr_feats.squeeze(1),
            'cxr_feats': cxr_feats.squeeze(1),
            'cls_token': cls_output,
            'attention_weights': attn_weights
        }
    
class TemporalAlignmentModule(nn.Module):
    def __init__(self, ehr_dim, cxr_dim, align_dim=128):
        super(TemporalAlignmentModule, self).__init__()
        
        self.ehr_dim = ehr_dim
        self.cxr_dim = cxr_dim
        self.align_dim = align_dim
        
        # Projection layers to common alignment space
        self.ehr_projection = nn.Linear(ehr_dim, align_dim)
        self.cxr_projection = nn.Linear(cxr_dim, align_dim)
        
        # Time-aware gating mechanism
        self.time_gate = nn.Sequential(
            nn.Linear(1, 64),  # Time difference input
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=align_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, ehr_feats, cxr_feats, time_diff=None):
        """
        ehr_feats: EHR features [batch_size, ehr_dim]
        cxr_feats: CXR features [batch_size, cxr_dim]
        time_diff: Time difference between EHR data collection and CXR acquisition [batch_size, 1]
        """
        # Project both modalities to common alignment space
        ehr_aligned = self.ehr_projection(ehr_feats)
        cxr_aligned = self.cxr_projection(cxr_feats)
        
        # Apply time-aware gating
        if time_diff is not None:
            time_weights = self.time_gate(time_diff)
            # Apply temporal weighting
            cxr_aligned = cxr_aligned * time_weights
        
        # Prepare for cross-attention
        ehr_aligned = ehr_aligned.unsqueeze(1)  # [batch_size, 1, align_dim]
        cxr_aligned = cxr_aligned.unsqueeze(1)  # [batch_size, 1, align_dim]
        
        # Apply cross-modal attention (EHR attending to CXR)
        aligned_output, attn_weights = self.cross_attention(
            ehr_aligned, cxr_aligned, cxr_aligned
        )
        
        return aligned_output.squeeze(1), attn_weights, time_weights if time_diff is not None else None
    
class CrossModalGenerator(nn.Module):
    def __init__(self, ehr_dim=256, cxr_dim=512, hidden_dim=512):
        super(CrossModalGenerator, self).__init__()
        
        # EHR -> CXR generator
        self.ehr_to_cxr = nn.Sequential(
            nn.Linear(ehr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cxr_dim)
        )
        
        # CXR -> EHR generator
        self.cxr_to_ehr = nn.Sequential(
            nn.Linear(cxr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ehr_dim)
        )
        
        # Reconstruction losses
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def forward(self, ehr_feats=None, cxr_feats=None):
        """
        Generates missing modality from available modality
        """
        losses = {}
        generated = {}
        
        # Generate CXR features from EHR features
        if ehr_feats is not None:
            generated_cxr = self.ehr_to_cxr(ehr_feats)
            generated['cxr_from_ehr'] = generated_cxr
            
            # Calculate reconstruction loss if real CXR features are available
            if cxr_feats is not None:
                losses['mse_cxr'] = self.mse_loss(generated_cxr, cxr_feats)
                target = torch.ones(ehr_feats.size(0)).to(ehr_feats.device)
                losses['cosine_cxr'] = self.cosine_loss(generated_cxr, cxr_feats, target)
        
        # Generate EHR features from CXR features
        if cxr_feats is not None:
            generated_ehr = self.cxr_to_ehr(cxr_feats)
            generated['ehr_from_cxr'] = generated_ehr
            
            # Calculate reconstruction loss if real EHR features are available
            if ehr_feats is not None:
                losses['mse_ehr'] = self.mse_loss(generated_ehr, ehr_feats)
                target = torch.ones(cxr_feats.size(0)).to(cxr_feats.device)
                losses['cosine_ehr'] = self.cosine_loss(generated_ehr, ehr_feats, target)
        
        return generated, losses