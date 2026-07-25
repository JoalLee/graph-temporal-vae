"""Simple MLP VAE baseline retained for ablation compatibility."""

import numpy as np
import torch
import torch.nn as nn

from .flows import RealNVP


class VanillaVAE(nn.Module):
    """MLP VAE without temporal convolutions or graph structure."""

    def __init__(self, input_dim, window_size, latent_dim, hidden_dims,
                 target_dim=None, chem_dim=31, psd_dim=230,
                 var_min=1e-4, var_max=10.0, heteroscedastic=True,
                 use_realnvp=False, realnvp_layers=4):
        super().__init__()
        self.use_realnvp = use_realnvp
        self.flow = (
            RealNVP(latent_dim, n_layers=realnvp_layers, hidden_dim=max(64, latent_dim))
            if use_realnvp else None
        )
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.target_dim = target_dim if target_dim is not None else input_dim
        self.chem_dim = chem_dim
        self.psd_dim = psd_dim
        self.var_min = var_min
        self.var_max = var_max
        self.heteroscedastic = heteroscedastic

        self.mask_embed = nn.Embedding(2, target_dim)
        nn.init.normal_(self.mask_embed.weight, mean=0.0, std=0.01)
        flat_dim = input_dim * window_size
        encoder_layers = []
        in_dim = flat_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.decoder = nn.Sequential(*decoder_layers)
        out_flat = self.target_dim * window_size
        self.recon_head = nn.Linear(hidden_dims[0], out_flat)
        if heteroscedastic:
            self.logvar_chem_head = nn.Linear(hidden_dims[0], chem_dim * window_size)
            self.logvar_psd_head = nn.Linear(hidden_dims[0], psd_dim * window_size)
        else:
            self.logvar_head = nn.Linear(hidden_dims[0], out_flat)

        self.last_graph_attention = None
        self.last_graph_attention_heads = None
        self.last_cross_modal_attention = None
        self.last_cross_modal_attention_heads = None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, cond=None, mask=None, aux_mask=None, sample_latent=True):
        batch, window, target_dim = x.shape
        embed_0 = self.mask_embed(torch.zeros(1, dtype=torch.long, device=x.device))
        embed_1 = self.mask_embed(torch.ones(1, dtype=torch.long, device=x.device))
        x_with_embed = x + embed_0 + mask.float() * (embed_1 - embed_0)
        x_with_embed = x_with_embed.transpose(1, 2)
        if cond is not None:
            cond = cond.transpose(1, 2)
            x_with_embed = torch.cat([x_with_embed, cond], dim=1)
        batch, channels, window = x_with_embed.shape
        x_flat = x_with_embed.view(batch, -1)
        expected_flat = self.input_dim * self.window_size
        if x_flat.shape[1] != expected_flat:
            raise RuntimeError(
                f"VanillaVAE forward() dimension mismatch!\n"
                f"  Model initialized with input_dim={self.input_dim} × window={self.window_size} = {expected_flat}\n"
                f"  Forward() received: C={channels} channels × W={window} window = {x_flat.shape[1]}\n"
                f"  target_dim={target_dim}, cond_dim={cond.shape[1] if cond is not None else 0}\n"
                "  Hint: Ensure total_aux_dim in train_ablation.py matches actual cond dimension"
            )
        hidden = self.encoder(x_flat)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        self.last_log_det_J = None
        self.last_z0 = z
        self.last_zK = z
        if self.use_realnvp:
            z, self.last_log_det_J = self.flow(z)
            self.last_zK = z
        decoded = self.decoder(z)
        recon_mu = self.recon_head(decoded).view(batch, self.target_dim, window)
        if self.heteroscedastic:
            logvar_chem = self.logvar_chem_head(decoded).view(batch, self.chem_dim, window)
            logvar_psd = self.logvar_psd_head(decoded).view(batch, self.psd_dim, window)
            recon_logvar = torch.cat([logvar_chem, logvar_psd], dim=1)
        else:
            recon_logvar = self.logvar_head(decoded).view(batch, self.target_dim, window)
        recon_logvar = torch.clamp(
            recon_logvar, np.log(self.var_min), np.log(self.var_max)
        )
        return (
            recon_mu.transpose(1, 2),
            recon_logvar.transpose(1, 2),
            mu,
            logvar,
            None,
        )

    def _enable_mc_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def compute_uncertainty(self, x, cond, mask, n_samples=50,
                            dist_type='gaussian', history=None,
                            return_extra_quantiles=False, return_samples=False,
                            enable_mc_dropout=True, sample_latent=True,
                            sample_likelihood=True):
        self.eval()
        if enable_mc_dropout:
            self._enable_mc_dropout()
        means = []
        logvars = []
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                recon_mean, recon_logvar = self.forward(
                    x, cond, mask, sample_latent=sample_latent
                )[:2]
                means.append(recon_mean)
                if recon_logvar is not None:
                    logvars.append(recon_logvar)
                    if not sample_likelihood:
                        sample = recon_mean
                    elif dist_type == 'student_t':
                        df = 3.0
                        variance = torch.exp(recon_logvar)
                        sigma = torch.sqrt(
                            (variance * (df - 2.0) / df).clamp(min=1e-10)
                        )
                        chi2 = torch.distributions.Chi2(df).sample(
                            recon_mean.shape
                        ).to(x.device)
                        sample = (
                            recon_mean
                            + sigma * torch.randn_like(recon_mean)
                            * torch.sqrt(df / chi2)
                        )
                    else:
                        sample = (
                            recon_mean
                            + torch.exp(0.5 * recon_logvar)
                            * torch.randn_like(recon_mean)
                        )
                    samples.append(sample)
                else:
                    samples.append(recon_mean)
        self.eval()
        means = torch.stack(means, dim=0)
        samples = torch.stack(samples, dim=0)
        epistemic_var = means.var(dim=0)
        pred_mean = means.mean(dim=0)
        epi_q05 = torch.quantile(means, 0.05, dim=0)
        epi_q95 = torch.quantile(means, 0.95, dim=0)
        epi_q025 = torch.quantile(means, 0.025, dim=0)
        epi_q975 = torch.quantile(means, 0.975, dim=0)
        pred_q05 = torch.quantile(samples, 0.05, dim=0)
        pred_q95 = torch.quantile(samples, 0.95, dim=0)
        pred_q025 = torch.quantile(samples, 0.025, dim=0)
        pred_q975 = torch.quantile(samples, 0.975, dim=0)
        aleatoric_var = (
            torch.exp(torch.stack(logvars, dim=0)).mean(dim=0)
            if logvars else None
        )
        total_var = epistemic_var + (
            aleatoric_var if aleatoric_var is not None else 0
        )
        result = (
            pred_mean, epistemic_var, aleatoric_var, total_var, None,
            pred_q05, pred_q95, epi_q05, epi_q95,
        )
        if return_extra_quantiles:
            result += (pred_q025, pred_q975, epi_q025, epi_q975)
        if return_samples:
            result += (samples, means)
        return result

    def get_learned_graph(self):
        return None

    def get_learned_graph_heads(self):
        return None

    def get_cross_modal_graph(self):
        return None

    def get_cross_modal_graph_heads(self):
        return None
