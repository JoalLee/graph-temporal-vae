"""
Uncertainty Quantification VAE Model
====================================
Extension of ImputationVAE with uncertainty quantification capabilities.
Supports:
- Phase 1: MC Dropout + Latent Sampling (Epistemic Uncertainty)
- Phase 2: Heteroscedastic Decoder (Aleatoric Uncertainty)
- Phase 3: Combined uncertainty output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
import numpy as np
from typing import Tuple, Optional

# Import base components from original model
from .model import TemporalBlock, AttentionPooling, Encoder


class DecoderUQ(nn.Module):
    """
    Uncertainty-Quantifying Decoder with Heteroscedastic Output.
    
    Outputs both mean and log-variance for each target dimension.
    
    Input: z [Batch, Latent_Dim], c [Batch, Window, Cond_Dim]
    Output: 
        - mean: [Batch, Window, Target_Dim]
        - log_variance: [Batch, Window, Target_Dim]
    """
    def __init__(self, latent_dim, cond_dim, num_channels, output_dim, kernel_size, dropout, window_size, dilations,
                 heteroscedastic=True):
        super(DecoderUQ, self).__init__()
        
        self.window_size = window_size
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.heteroscedastic = heteroscedastic
        
        # Project broadcasted z + time-series c to hidden channels
        self.input_conv = nn.Conv1d(latent_dim + cond_dim, num_channels, kernel_size=1)
        
        layers = []
        num_levels = len(dilations)
        
        for i in range(num_levels):
            dilation_size = dilations[i]
            pad = (kernel_size - 1) * dilation_size // 2
            
            layers.append(
                TemporalBlock(
                    n_inputs=num_channels, 
                    n_outputs=num_channels, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    dilation=dilation_size, 
                    padding=pad, 
                    dropout=dropout
                )
            )
            
        self.tcn = nn.Sequential(*layers)
        
        # Mean output head
        self.final_conv_mean = nn.Conv1d(num_channels, output_dim, kernel_size=1)
        
        # Variance output head (only if heteroscedastic)
        if heteroscedastic:
            self.final_conv_logvar = nn.Conv1d(num_channels, output_dim, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.final_conv_mean.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.final_conv_mean.bias, 0)
        
        if self.heteroscedastic:
            nn.init.kaiming_normal_(self.final_conv_logvar.weight, mode='fan_in', nonlinearity='linear')
            # Initialize log-variance to small value (variance ~ 0.1)
            nn.init.constant_(self.final_conv_logvar.bias, -2.0)

    def forward(self, z, c):
        # z: [Batch, Latent_Dim] - Global latent vector
        # c: [Batch, Window, Cond_Dim] - Time-series condition features
        
        batch_size = z.size(0)
        
        # Broadcast z to window length
        z_broadcast = z.unsqueeze(-1).expand(-1, -1, self.window_size)
        
        # Permute c for Conv1d
        c_permuted = c.permute(0, 2, 1)
        
        # Concatenate
        z_cond = torch.cat([z_broadcast, c_permuted], dim=1)
        
        # Project and refine
        out = self.input_conv(z_cond)
        out = self.tcn(out)
        
        # Mean output
        mean = self.final_conv_mean(out)  # [Batch, Target, Window]
        
        if self.heteroscedastic:
            # Log-variance output
            logvar = self.final_conv_logvar(out)  # [Batch, Target, Window]
            # Clamp log-variance for numerical stability
            logvar = torch.clamp(logvar, min=-10, max=10)
            return mean, logvar
        else:
            return mean, None


class ImputationVAE_UQ(nn.Module):
    """
    Uncertainty-Quantifying Conditional VAE for Time Series Imputation.
    
    Extensions over base ImputationVAE:
    1. MC Dropout: Keep dropout active during inference
    2. Latent Sampling: Always sample z (not just during training)
    3. Heteroscedastic Output: Predict both mean and variance
    
    Architecture:
        Input: x [Batch, Window, Target], c [Batch, Window, Aux], mask [Batch, Window, Target]
        Encoder: TCN -> Attention Pool -> [Batch, Latent_Dim]
        Decoder: Broadcast z + concat c -> TCN -> [Batch, Window, Target] (mean, logvar)
    """
    def __init__(self, 
                 target_dim, 
                 aux_dim, 
                 window_size, 
                 latent_dim=256, 
                 hidden_dims=[256, 256, 256],
                 encoder_layers=6,
                 decoder_layers=6,
                 heteroscedastic=True,
                 dropout=0.2): 
        super(ImputationVAE_UQ, self).__init__()
        
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.heteroscedastic = heteroscedastic
        self._mc_dropout_enabled = False
        
        # Config for TCN
        tcn_channels = hidden_dims[0] if hidden_dims else 256
        kernel_size = 3
        
        # Dynamically generate dilations
        enc_dilations = [2**i for i in range(encoder_layers)]
        dec_dilations = [2**i for i in range(decoder_layers)]
        
        # Input to encoder is [X, C, M]
        input_dim = target_dim + aux_dim + target_dim
        
        self.encoder = Encoder(
            input_dim=input_dim, 
            num_channels=tcn_channels, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            latent_dim=latent_dim, 
            window_size=window_size,
            dilations=enc_dilations
        )
        
        # UQ Decoder (heteroscedastic)
        self.decoder = DecoderUQ(
            latent_dim=latent_dim,
            cond_dim=aux_dim,
            num_channels=tcn_channels, 
            output_dim=target_dim, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            window_size=window_size,
            dilations=dec_dilations,
            heteroscedastic=heteroscedastic
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling z."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation during inference."""
        self._mc_dropout_enabled = True
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def disable_mc_dropout(self):
        """Disable MC Dropout (normal inference mode)."""
        self._mc_dropout_enabled = False
        self.eval()

    def forward(self, x, c, mask, force_sample=False):
        """
        Forward pass with optional stochastic sampling.
        
        Args:
            x: [Batch, Window, Target] - Input features (masked)
            c: [Batch, Window, Aux] - Condition features
            mask: [Batch, Window, Target] - Observation mask (1=observed)
            force_sample: If True, always sample z (for MC inference)
            
        Returns:
            recon_mean: [Batch, Window, Target] - Reconstructed mean
            recon_logvar: [Batch, Window, Target] or None - Log-variance (if heteroscedastic)
            mu: [Batch, Latent_Dim] - Latent mean
            logvar: [Batch, Latent_Dim] - Latent log-variance
        """
        # Concatenate inputs
        inputs = torch.cat([x, c, mask], dim=-1)
        inputs = inputs.permute(0, 2, 1)  # [Batch, Channels, Window]
        
        # Encode to global latent
        mu, logvar = self.encoder(inputs)
        
        # Sampling decision
        if self.training or force_sample or self._mc_dropout_enabled:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        # Decode with condition
        recon_mean, recon_logvar = self.decoder(z, c)
        
        # Permute back: [Batch, Window, Target]
        recon_mean = recon_mean.permute(0, 2, 1)
        if recon_logvar is not None:
            recon_logvar = recon_logvar.permute(0, 2, 1)
        
        return recon_mean, recon_logvar, mu, logvar

    @torch.no_grad()
    def sample_predictions(self, x, c, mask, n_samples=50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate N stochastic predictions for epistemic uncertainty estimation.
        
        Uses both MC Dropout and latent space sampling.
        Optimized with batched sampling where possible.
        
        Args:
            x, c, mask: Input tensors [Batch, Window, Target/Aux]
            n_samples: Number of forward passes
            
        Returns:
            means: [N, Batch, Window, Target] - Stacked mean predictions
            logvars: [N, Batch, Window, Target] or None - Stacked log-variance predictions
        """
        self.enable_mc_dropout()
        
        batch_size = x.size(0)
        
        # Pre-allocate tensors for efficiency
        means = torch.zeros(n_samples, batch_size, self.window_size, self.target_dim, device=x.device)
        logvars = torch.zeros(n_samples, batch_size, self.window_size, self.target_dim, device=x.device) if self.heteroscedastic else None
        
        # Encode once to get mu, logvar for latent sampling
        inputs = torch.cat([x, c, mask], dim=-1)
        inputs_perm = inputs.permute(0, 2, 1)
        mu_z, logvar_z = self.encoder(inputs_perm)
        
        for i in range(n_samples):
            # Sample z from latent distribution
            z = self.reparameterize(mu_z, logvar_z)
            
            # Decode
            recon_mean, recon_logvar = self.decoder(z, c)
            means[i] = recon_mean.permute(0, 2, 1)
            if logvars is not None and recon_logvar is not None:
                logvars[i] = recon_logvar.permute(0, 2, 1)
        
        self.disable_mc_dropout()
        
        return means, logvars

    def compute_uncertainty(self, x, c, mask, n_samples=50):
        """
        Compute total uncertainty by combining epistemic and aleatoric components.
        
        Returns:
            pred_mean: [Batch, Window, Target] - Mean prediction (averaged over samples)
            epistemic_var: [Batch, Window, Target] - Variance from model uncertainty
            aleatoric_var: [Batch, Window, Target] or None - Variance from data noise
            total_var: [Batch, Window, Target] - Combined variance (Law of Total Variance)
        """
        # Get N stochastic predictions
        sampled_means, sampled_logvars = self.sample_predictions(x, c, mask, n_samples)
        
        # Prediction Intervals (Percentiles)
        # Epistemic-only from sampled_means
        epi_q05 = torch.quantile(sampled_means, 0.05, dim=0)
        epi_q95 = torch.quantile(sampled_means, 0.95, dim=0)
        
        # Total Prediction Intervals (Generative)
        # Note: sampled_means are p(y|x, z), logvars are p(y|x, z). 
        # For a full generative PI, one would need to sample from the distributions like in model_graph_uq.py.
        # Since this model_uq.py is a base version, we provide the means-only percentiles 
        # as a fallback if no actual generative samples are drawn.
        # But to be consistent with model_graph_uq.py, we'll return the same structure.
        total_q05 = epi_q05 # Fallback
        total_q95 = epi_q95 # Fallback
        
        return pred_mean, epistemic_var, aleatoric_var, total_var, None, total_q05, total_q95, epi_q05, epi_q95

    @torch.no_grad()
    def compute_uncertainty_physical(self, x, c, mask, n_samples=50, 
                                      scaler=None, size_indices=None):
        """
        Compute uncertainty directly in physical space for log-transformed data.
        
        For SMPS/APS data (log1p transformed), calculate variance AFTER applying
        inverse_transform and expm1 for accurate physical-space uncertainty.
        
        Args:
            x, c, mask: Input tensors
            n_samples: Number of MC samples
            scaler: sklearn StandardScaler for inverse transform
            size_indices: List of indices for log-transformed features (SMPS/APS)
            
        Returns:
            pred_mean_physical: Mean prediction in physical space
            epistemic_std_physical: Epistemic std in physical space
            aleatoric_var_scaled: Aleatoric variance in scaled space (for CHEM)
        """
        sampled_means, sampled_logvars = self.sample_predictions(x, c, mask, n_samples)
        # sampled_means: [N, Batch, Window, Target]
        
        n_samples_actual, batch_size, window_size, target_dim = sampled_means.shape
        
        # Convert to numpy for scaler operations
        sampled_means_np = sampled_means.cpu().numpy()
        
        # Apply inverse transform to each sample
        means_physical = np.zeros_like(sampled_means_np)
        for i in range(n_samples_actual):
            # Reshape for scaler: [Batch*Window, Target]
            reshaped = sampled_means_np[i].reshape(-1, target_dim)
            if scaler is not None:
                reshaped = scaler.inverse_transform(reshaped)
            
            # Apply expm1 to size bins
            if size_indices is not None and len(size_indices) > 0:
                reshaped[:, size_indices] = np.expm1(reshaped[:, size_indices])
            
            means_physical[i] = reshaped.reshape(batch_size, window_size, target_dim)
        
        # Calculate statistics in physical space
        pred_mean_physical = means_physical.mean(axis=0)  # [Batch, Window, Target]
        epistemic_std_physical = means_physical.std(axis=0)  # [Batch, Window, Target]
        
        # Aleatoric variance (keep in scaled space for now)
        aleatoric_var_scaled = None
        if sampled_logvars is not None:
            aleatoric_var_scaled = torch.exp(sampled_logvars).mean(dim=0).cpu().numpy()
        
        return pred_mean_physical, epistemic_std_physical, aleatoric_var_scaled


# Compatibility wrapper for loading original model weights
def load_from_base_model(uq_model: ImputationVAE_UQ, base_weights_path: str, device='cpu'):
    """
    Load weights from original ImputationVAE into UQ model.
    
    Only loads encoder and decoder.mean weights (skips decoder.logvar).
    """
    base_state = torch.load(base_weights_path, map_location=device)
    
    # Create mapping for compatible keys
    uq_state = uq_model.state_dict()
    
    for key, value in base_state.items():
        # Map decoder.final_conv -> decoder.final_conv_mean
        if 'decoder.final_conv' in key:
            new_key = key.replace('decoder.final_conv', 'decoder.final_conv_mean')
            if new_key in uq_state:
                uq_state[new_key] = value
        elif key in uq_state:
            uq_state[key] = value
    
    uq_model.load_state_dict(uq_state, strict=False)
    print(f"✅ Loaded base model weights from {base_weights_path}")
    print(f"   Note: decoder.final_conv_logvar initialized with defaults")
    
    return uq_model
