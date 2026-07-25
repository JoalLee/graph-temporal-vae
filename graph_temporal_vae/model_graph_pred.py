"""
Graph-Enhanced VAE for Multi-Step Prediction
=============================================
Predicts future time steps using Graph Attention and TCN architecture.
Based on ImputationVAE_Graph but modified for forecasting.

Input: X[1:T] (historical window)
Output: X[T+1:T+H] (future predictions with uncertainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as param
import numpy as np


class InputGraphLayer(nn.Module):
    """
    Feature-space attention for learning species relationships.
    Same as imputation version.
    """
    
    def __init__(self, n_features, window_size, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(window_size, window_size)
        self.k_proj = nn.Linear(window_size, window_size)
        self.v_proj = nn.Linear(window_size, window_size)
        self.out_proj = nn.Linear(window_size, window_size)
        
        self.layer_norm = nn.LayerNorm(window_size)
        self.dropout = nn.Dropout(dropout)
        
        self.last_attention_weights = None
    
    def forward(self, x, obs_mask=None):
        """
        Args:
            x: [Batch, Channels, Window]
            obs_mask: [Batch, Window, Channels] - observation mask
        """
        B, C, W = x.shape
        
        if obs_mask is None:
            mask = torch.ones(B, C, W, device=x.device)
        else:
            mask = obs_mask.permute(0, 2, 1).float()
        
        # Masked normalization
        x_masked = x * mask
        obs_count = mask.sum(dim=2, keepdim=True).clamp(min=1)
        x_mean = x_masked.sum(dim=2, keepdim=True) / obs_count
        x_centered = (x - x_mean) * mask
        x_var = (x_centered ** 2).sum(dim=2, keepdim=True) / obs_count.clamp(min=2)
        x_std = torch.sqrt(x_var + 1e-8)
        x_normed = (x_centered / x_std) * mask
        
        # Attention computation
        Q = self.q_proj(x_normed)
        K = self.k_proj(x_normed)
        V = self.v_proj(x_normed)
        
        Q_masked = Q * mask
        K_masked = K * mask
        
        scale = np.sqrt(W)
        attn_logits = torch.bmm(Q_masked, K_masked.transpose(1, 2)) / scale
        
        count_i = mask.sum(dim=2, keepdim=True)
        count_j = mask.sum(dim=2, keepdim=True).transpose(1, 2)
        pair_count_approx = torch.sqrt(count_i * count_j + 1e-8)
        attn_logits = attn_logits * (np.sqrt(W) / pair_count_approx.clamp(min=1))
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn_weights)
        
        self.last_attention_weights = attn_weights.detach().mean(dim=0)
        
        out = torch.bmm(attn, V)
        out = self.out_proj(out)
        out = self.layer_norm(out + x)
        
        return out, attn_weights


class PredictionEncoder(nn.Module):
    """
    Encoder for prediction: processes historical window.
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim, window_size, 
                 num_layers=5, kernel_size=3, dropout=0.1, n_graph_heads=4):
        super().__init__()
        
        self.input_dim = input_dim
        self.window_size = window_size
        
        # Input Graph Layer
        self.input_graph_layer = InputGraphLayer(
            n_features=input_dim,
            window_size=window_size,
            n_heads=n_graph_heads,
            dropout=dropout
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dims[0], 1)
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dims[min(i, len(hidden_dims)-1)]
            out_ch = hidden_dims[min(i+1, len(hidden_dims)-1)] if i < num_layers-1 else hidden_dims[-1]
            dilation = 2 ** i
            
            self.tcn_layers.append(
                nn.Sequential(
                    param.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                                padding=(kernel_size-1)*dilation//2,
                                                dilation=dilation)),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        self.last_input_graph_attention = None
    
    def forward(self, x, obs_mask=None):
        """
        Args:
            x: [B, input_dim, window_size]
            obs_mask: [B, window_size, input_dim]
        """
        # Graph attention
        x_enhanced, attn_weights = self.input_graph_layer(x, obs_mask)
        self.last_input_graph_attention = attn_weights.detach().mean(dim=0)
        
        # TCN encoding
        h = self.input_proj(x_enhanced)
        for layer in self.tcn_layers:
            h_new = layer(h)
            if h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        
        # Latent projection
        h_pooled = self.avg_pool(h).squeeze(-1)
        mu = self.fc_mu(h_pooled)
        logvar = self.fc_logvar(h_pooled)
        
        return mu, logvar, attn_weights


class PredictionDecoder(nn.Module):
    """
    Decoder for prediction: generates future time steps.
    
    Key difference from imputation decoder:
    - Output window size (H) can be different from input window size (T)
    """
    
    def __init__(self, latent_dim, hidden_dims, output_dim, forecast_horizon,
                 num_layers=5, kernel_size=3, dropout=0.1, heteroscedastic=True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.heteroscedastic = heteroscedastic
        
        # Project latent to initial hidden state
        self.z_proj = nn.Linear(latent_dim, hidden_dims[0] * forecast_horizon)
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_dims[min(i, len(hidden_dims)-1)]
            out_ch = hidden_dims[min(i+1, len(hidden_dims)-1)] if i < num_layers-1 else hidden_dims[-1]
            dilation = 2 ** i
            
            self.tcn_layers.append(
                nn.Sequential(
                    param.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                                padding=(kernel_size-1)*dilation//2,
                                                dilation=dilation)),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output projections
        self.output_proj = nn.Conv1d(hidden_dims[-1], output_dim, 1)
        
        if heteroscedastic:
            self.logvar_proj = nn.Conv1d(hidden_dims[-1], output_dim, 1)
    
    def forward(self, z):
        """
        Args:
            z: [B, latent_dim]
            
        Returns:
            mean: [B, output_dim, forecast_horizon]
            logvar: [B, output_dim, forecast_horizon] or None
        """
        B = z.shape[0]
        
        # Project to sequence
        h = self.z_proj(z)
        h = h.view(B, -1, self.forecast_horizon)  # [B, hidden, H]
        
        # TCN decoding
        for layer in self.tcn_layers:
            h_new = layer(h)
            if h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new
        
        # Output projection
        mean = self.output_proj(h)  # [B, output_dim, H]
        
        logvar = None
        if self.heteroscedastic:
            logvar = self.logvar_proj(h)
            logvar = torch.clamp(logvar, min=-4.6, max=10)
        
        return mean, logvar


class PredictionVAE_Graph(nn.Module):
    """
    Graph-Enhanced VAE for Multi-Step Prediction.
    
    Input: Historical window [B, T, D]
    Output: Future predictions [B, H, D] with uncertainty
    """
    
    def __init__(self, target_dim, aux_dim, latent_dim, hidden_dims,
                 input_window, forecast_horizon=12,
                 encoder_layers=5, decoder_layers=5, 
                 kernel_size=3, dropout=0.1, n_graph_heads=4,
                 heteroscedastic=True):
        super().__init__()
        
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        
        # Total input: target + aux + mask indicator
        input_dim = target_dim + aux_dim + target_dim
        
        # Encoder
        self.encoder = PredictionEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            window_size=input_window,
            num_layers=encoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            n_graph_heads=n_graph_heads
        )
        
        # Decoder
        self.decoder = PredictionDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=target_dim,
            forecast_horizon=forecast_horizon,
            num_layers=decoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            heteroscedastic=heteroscedastic
        )
        
        self.last_graph_attention = None
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, cond, mask):
        """
        Args:
            x: [B, T, target_dim] - Historical target features
            cond: [B, T, aux_dim] - Conditioning features
            mask: [B, T, target_dim] - Observation mask
            
        Returns:
            pred_mean: [B, H, target_dim] - Future predictions
            pred_logvar: [B, H, target_dim] - Uncertainty
            mu: [B, latent_dim]
            logvar: [B, latent_dim]
            graph_attention: [B, C, C]
        """
        B, T, D = x.shape
        
        # Prepare encoder input
        inputs = torch.cat([x, cond, mask], dim=-1)
        inputs = inputs.permute(0, 2, 1)  # [B, input_dim, T]
        
        # Create observation mask for graph layer
        target_obs_mask = mask
        cond_dim = cond.shape[-1]
        full_obs_mask = torch.cat([
            target_obs_mask,
            torch.ones(B, T, cond_dim, device=x.device),
            torch.ones(B, T, self.target_dim, device=x.device)
        ], dim=-1)
        
        # Encode
        mu, logvar, graph_attention = self.encoder(inputs, full_obs_mask)
        self.last_graph_attention = graph_attention.detach().mean(dim=0)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode to future
        pred_mean, pred_logvar = self.decoder(z)
        
        # Reshape: [B, D, H] -> [B, H, D]
        pred_mean = pred_mean.permute(0, 2, 1)
        if pred_logvar is not None:
            pred_logvar = pred_logvar.permute(0, 2, 1)
        
        return pred_mean, pred_logvar, mu, logvar, graph_attention
