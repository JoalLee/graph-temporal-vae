"""Reference trainer: fit an ImputationVAE_Graph on a CSV and save a
self-contained checkpoint bundle.

This is intentionally smaller than the full research training loop
(no wandb and no dozens of ablation-diagnostic metrics)
-- it exists so a new dataset can be trained on without leaving this
package. The bundle it saves carries everything `infer.impute` needs
(weights, model kwargs, column names, and the *training-fit* normalization
stats), so there is no separate config file to keep in sync by hand.
"""
import copy
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import (
    NaNAwareStandardScaler,
    WindowedTimeSeriesDataset,
    chronological_split_index,
    load_frame,
    sample_dynamic_heldout_mask,
)
from .model_graph_uq import ImputationVAE_Graph
from .utils import KLAnnealingScheduler, LRWarmupCosineScheduler, setup_device


def vae_loss(recon_mean, recon_logvar, target, obs_mask, mu, logvar, beta, metric_mask=None):
    obs_mask = obs_mask.float()
    metric_mask = obs_mask if metric_mask is None else metric_mask.float()
    n_obs = metric_mask.sum().clamp(min=1.0)

    if recon_logvar is not None:
        var = torch.exp(recon_logvar).clamp(min=1e-6)
        nll = 0.5 * (recon_logvar + (target - recon_mean) ** 2 / var)
        recon = (nll * metric_mask).sum() / n_obs
    else:
        recon = ((recon_mean - target) ** 2 * metric_mask).sum() / n_obs

    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    loss = recon + beta * kl
    return loss, recon.detach(), kl.detach()


def empirical_crps(samples, target, mask):
    """Compute empirical CRPS on selected entries of MC predictive samples."""
    mask = mask.float()
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=target.device)
    first = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    pairwise = 0.5 * (samples[:, None] - samples[None, :]).abs().mean(dim=(0, 1))
    score = first - pairwise
    return (score * mask).sum() / mask.sum()


def masked_mse(prediction, target, mask):
    """Mean squared error over a held-out mask."""
    mask = mask.float()
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def masked_nll(recon_mean, recon_logvar, target, mask):
    """Gaussian reconstruction NLL over a held-out mask."""
    if recon_logvar is None:
        return masked_mse(recon_mean, target, mask)
    variance = torch.exp(recon_logvar).clamp(min=1e-6)
    nll = 0.5 * (recon_logvar + (target - recon_mean) ** 2 / variance)
    mask = mask.float()
    return (nll * mask).sum() / mask.sum().clamp(min=1.0)


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainConfig, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        warmup_epochs = config.kl_warmup_epochs or max(1, int(config.epochs * 0.2))
        self.kl_scheduler = KLAnnealingScheduler(
            config.epochs, warmup_epochs, max_beta=config.kl_max_beta
        )
        self.lr_scheduler = LRWarmupCosineScheduler(
            self.optimizer, config.epochs, max(1, int(config.epochs * 0.05))
        )

    def _run_epoch(self, loader, epoch, train):
        self.model.train(train)
        beta = self.kl_scheduler.get_beta(epoch)
        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(train):
            for batch in tqdm(loader, desc=f"{'train' if train else 'train'} epoch {epoch + 1}", leave=False):
                input_x = batch["input_x"].to(self.device)
                cond = batch["cond"].to(self.device)
                input_mask = batch["input_mask"].to(self.device)
                target = batch["target"].to(self.device)
                obs_mask = batch["obs_mask"].to(self.device)

                recon_mean, recon_logvar, mu, logvar, _ = self.model(input_x, cond, input_mask)
                loss, _recon, _kl = vae_loss(recon_mean, recon_logvar, target, obs_mask, mu, logvar, beta)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _run_validation(self, loader, epoch):
        """Evaluate only on the fixed selection-HO positions."""
        self.model.eval()
        total_nll = 0.0
        total_mse = 0.0
        total_crps = 0.0
        n_batches = 0
        n_crps_batches = 0
        compute_crps = (
            self.config.validation_metric == "ho_crps"
            and (epoch == 0 or (epoch + 1) % max(1, self.config.val_crps_every_n_epochs) == 0)
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"val epoch {epoch + 1}", leave=False):
                input_x = batch["input_x"].to(self.device)
                cond = batch["cond"].to(self.device)
                input_mask = batch["input_mask"].to(self.device)
                target = batch["target"].to(self.device)
                obs_mask = batch["obs_mask"].to(self.device)
                heldout_mask = batch["heldout_mask"].to(self.device)

                recon_mean, recon_logvar, mu, logvar, _ = self.model(input_x, cond, input_mask)
                ho_nll = masked_nll(recon_mean, recon_logvar, target, heldout_mask)
                ho_mse = masked_mse(recon_mean, target, heldout_mask)
                total_nll += ho_nll.item()
                total_mse += ho_mse.item()
                n_batches += 1

                if compute_crps:
                    result = self.model.compute_uncertainty(
                        input_x,
                        cond,
                        input_mask,
                        n_samples=max(2, self.config.val_crps_mc_samples),
                        return_samples=True,
                    )
                    samples = result[-2]
                    crps = empirical_crps(samples, target, heldout_mask)
                    if torch.isfinite(crps):
                        total_crps += crps.item()
                        n_crps_batches += 1

        return {
            "ho_nll": total_nll / max(n_batches, 1),
            "ho_mse": total_mse / max(n_batches, 1),
            "ho_crps": (total_crps / n_crps_batches) if n_crps_batches else None,
        }

    def fit(self):
        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            self.lr_scheduler.step(epoch)
            train_loss = self._run_epoch(self.train_loader, epoch, train=True)

            if self.val_loader is not None:
                val_metrics = self._run_validation(self.val_loader, epoch)
                metric = val_metrics.get(self.config.validation_metric)
                # CRPS may be intentionally evaluated every N epochs. Do not
                # count skipped epochs against patience.
                if metric is None:
                    metric = None
                print(
                    f"epoch {epoch + 1}/{self.config.epochs}  train_loss={train_loss:.4f} "
                    f"val_ho_nll={val_metrics['ho_nll']:.4f} "
                    f"val_ho_mse={val_metrics['ho_mse']:.4f} "
                    f"val_ho_crps={val_metrics['ho_crps'] if val_metrics['ho_crps'] is not None else 'skipped'}"
                )
            else:
                val_metrics = {"ho_nll": train_loss, "ho_mse": train_loss, "ho_crps": None}
                metric = train_loss

            if metric is not None and metric < best_val - 1e-6:
                best_val = metric
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            elif metric is not None:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    print(f"early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val


def train_from_config(config: TrainConfig, save_path: str) -> float:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    frame = load_frame(config.csv, config.timestamp_col, config.target_cols, config.aux_cols)

    target_raw = frame[config.target_cols].to_numpy(dtype=np.float64)
    aux_raw = (
        frame[config.aux_cols].to_numpy(dtype=np.float64)
        if config.aux_cols
        else np.zeros((len(frame), 0))
    )

    split_idx = chronological_split_index(len(frame), config.val_fraction)
    train_target, val_target = target_raw[:split_idx], target_raw[split_idx:]
    train_aux, val_aux = aux_raw[:split_idx], aux_raw[split_idx:]
    train_aux_mask = ~np.isnan(train_aux)
    val_aux_mask = ~np.isnan(val_aux)

    scaler_target = NaNAwareStandardScaler().fit(train_target)
    scaler_aux = NaNAwareStandardScaler().fit(train_aux) if config.aux_cols else NaNAwareStandardScaler().fit(
        np.zeros((1, 0))
    )

    train_target_scaled = scaler_target.transform(train_target)
    train_target_scaled[np.isnan(train_target)] = np.nan  # keep NaN as the missingness signal
    val_target_scaled = scaler_target.transform(val_target)
    val_target_scaled[np.isnan(val_target)] = np.nan

    train_aux_scaled = scaler_aux.transform(train_aux) if config.aux_cols else train_aux
    val_aux_scaled = scaler_aux.transform(val_aux) if config.aux_cols else val_aux

    n_chem = int(config.model_kwargs.get("n_chem", 0))
    dynamic_mask_config = {
        "target_ratio": config.dynamic_mask_target_ratio,
        "mean_duration": config.dynamic_mask_mean_duration,
        "std_duration": config.dynamic_mask_std_duration,
        "min_duration": config.dynamic_mask_min_duration,
        "max_duration": config.dynamic_mask_max_duration,
        "chem_blocks": config.dynamic_mask_chem_blocks,
        "psd_blocks": config.dynamic_mask_psd_blocks,
        "n_chem": n_chem,
    }
    val_selection_mask = sample_dynamic_heldout_mask(
        ~np.isnan(val_target),
        {**dynamic_mask_config, "ensure_nonempty": True},
        seed=config.selection_val_seed,
    ) if len(val_target) else None
    if len(val_target) and val_selection_mask.sum() == 0:
        raise ValueError("Validation split has no observed target positions for selection-HO validation")

    train_dataset = WindowedTimeSeriesDataset(
        train_target_scaled, train_aux_scaled, config.window_size, config.stride,
        mode="train", denoise_prob=config.denoise_prob, seed=config.seed,
        aux_mask=train_aux_mask, aux_mask_channel=True,
        dynamic_mask_config=dynamic_mask_config,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training split ({len(train_target)} rows) is shorter than window_size={config.window_size}"
        )

    val_dataset = WindowedTimeSeriesDataset(
        val_target_scaled, val_aux_scaled, config.window_size, config.stride,
        mode="val", seed=config.seed, aux_mask=val_aux_mask,
        aux_mask_channel=True, selection_mask=val_selection_mask,
    )
    if len(val_target) and config.val_fraction > 0 and len(val_dataset) == 0:
        raise ValueError(
            f"Validation split ({len(val_target)} rows) is shorter than window_size={config.window_size}"
        )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        if len(val_dataset) > 0
        else None
    )

    model_kwargs = dict(config.model_kwargs)
    model = ImputationVAE_Graph(
        target_dim=len(config.target_cols),
        aux_dim=2 * len(config.aux_cols),
        window_size=config.window_size,
        **model_kwargs,
    )

    device = setup_device()
    trainer = Trainer(model, train_loader, val_loader, config, device)
    best_val = trainer.fit()

    bundle = {
        "bundle_version": 2,
        "state_dict": trainer.model.state_dict(),
        "model_kwargs": model_kwargs,
        "target_cols": list(config.target_cols),
        "aux_cols": list(config.aux_cols),
        "window_size": config.window_size,
        "aux_missing_mode": "mask_channel",
        "aux_mask_channel": True,
        "selection_mask_protocol": "fixed_dynamic_ho",
        "schema": {
            "target_cols": list(config.target_cols),
            "aux_value_cols": list(config.aux_cols),
            "aux_mask_cols": [f"{col}__observed" for col in config.aux_cols],
            "target_dim": len(config.target_cols),
            "cond_dim": 2 * len(config.aux_cols),
        },
        "scaler_target": scaler_target.to_dict(),
        "scaler_aux": scaler_aux.to_dict(),
        "config": config.to_dict(),
    }
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(bundle, save_path)

    return best_val
