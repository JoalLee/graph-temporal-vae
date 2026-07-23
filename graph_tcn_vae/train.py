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
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from .config import TrainConfig
from .data import (
    NaNAwareStandardScaler,
    WindowedTimeSeriesDataset,
    chronological_split_index,
    load_frame,
    sample_anchor_constrained_heldout_mask,
    sample_dynamic_heldout_mask,
    transform_target_values,
)
from .model_graph_uq import ImputationVAE_Graph
from .utils import KLAnnealingScheduler, LRWarmupCosineScheduler, is_interactive, setup_device


def _seed_window_worker(_worker_id):
    """Give each DataLoader worker an independent deterministic mask stream."""
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2**32))


def _student_t_nll(recon_mean, recon_logvar, target, var_min=1e-3, var_max=10.0, df=None):
    log_variance = recon_logvar.clamp(min=np.log(var_min), max=np.log(var_max))
    variance = torch.exp(log_variance)
    if df is None:
        df = torch.tensor(3.0, device=recon_mean.device, dtype=recon_mean.dtype)
    df = torch.as_tensor(df, device=recon_mean.device, dtype=recon_mean.dtype)
    while df.ndim < recon_mean.ndim:
        df = df.unsqueeze(0)
    sigma_sq = (variance * (df - 2.0) / df).clamp(min=1e-10)
    const = (
        torch.lgamma((df + 1.0) / 2.0)
        - torch.lgamma(df / 2.0)
        - 0.5 * torch.log(df * torch.tensor(np.pi, device=recon_mean.device, dtype=recon_mean.dtype))
    )
    residual_sq = (target - recon_mean).square()
    return -const + 0.5 * torch.log(sigma_sq) + ((df + 1.0) / 2.0) * torch.log1p(
        residual_sq / (df * sigma_sq)
    )


def _reduce_window_feature_loss(point_loss, mask, normalization, n_chem=0,
                                use_family_balanced_loss=False,
                                family_loss_chem_weight=0.5,
                                family_loss_scale="target_dim",
                                chem_feature_weight=1.0,
                                psd_feature_weight=1.0):
    mask = mask.float()
    n_features = point_loss.shape[-1]
    feature_weights = torch.full(
        (n_features,), float(psd_feature_weight),
        device=point_loss.device, dtype=point_loss.dtype,
    )
    if n_chem > 0:
        feature_weights[:min(n_chem, n_features)] = float(chem_feature_weight)
    weighted_point_loss = point_loss * feature_weights.view(1, 1, -1)
    if normalization == "observed_mean":
        return (weighted_point_loss * mask).sum() / mask.sum().clamp(min=1.0)

    window_size = point_loss.shape[1]
    loss_per_feature = (weighted_point_loss * mask).sum(dim=1)
    count_per_feature = mask.sum(dim=1).clamp(min=1.0)
    normalized = loss_per_feature * (window_size / count_per_feature)
    if use_family_balanced_loss and 0 < n_chem < normalized.shape[-1]:
        chem = normalized[:, :n_chem].mean(dim=1)
        psd = normalized[:, n_chem:].mean(dim=1)
        reduced = family_loss_chem_weight * chem + (1.0 - family_loss_chem_weight) * psd
        if family_loss_scale == "target_dim":
            reduced = reduced * normalized.shape[-1]
        return reduced.mean()
    return normalized.sum(dim=1).mean()


def _latent_kl_loss(mu, logvar, model=None, prior_type="gaussian"):
    mu = torch.clamp(mu, min=-100.0, max=100.0)
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    if model is not None and getattr(model, "last_log_det_J", None) is not None:
        z0 = model.last_z0
        z_k = model.last_zK
        log_det = model.last_log_det_J
        log_q_z0 = -0.5 * (
            np.log(2.0 * np.pi) + logvar + (z0 - mu).square() / logvar.exp()
        ).sum(dim=1)
        if prior_type == "student_t":
            df_getter = getattr(model, "get_prior_df", None)
            df = df_getter(device=z_k.device, dtype=z_k.dtype) if callable(df_getter) else 3.0
            df = torch.as_tensor(df, device=z_k.device, dtype=z_k.dtype)
            const = (
                torch.lgamma((df + 1.0) / 2.0)
                - torch.lgamma(df / 2.0)
                - 0.5 * torch.log(df * torch.tensor(np.pi, device=z_k.device, dtype=z_k.dtype))
            )
            log_p = const - ((df + 1.0) / 2.0) * torch.log1p(z_k.square() / df)
        elif prior_type == "laplace":
            log_p = -(np.log(2.0) + z_k.abs())
        else:
            log_p = -0.5 * (np.log(2.0 * np.pi) + z_k.square())
        return (log_q_z0 - log_det - log_p.sum(dim=1)).mean()
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1))


def vae_loss(recon_mean, recon_logvar, target, obs_mask, mu, logvar, beta, metric_mask=None,
             *, model=None, prior_type="gaussian", use_student_t_nll=False,
             loss_normalization="observed_mean", n_chem=0,
             use_family_balanced_loss=False, family_loss_chem_weight=0.5,
             family_loss_scale="target_dim", chem_feature_weight=1.0,
             psd_feature_weight=1.0, var_min=1e-3, var_max=10.0):
    obs_mask = obs_mask.float()
    metric_mask = obs_mask if metric_mask is None else metric_mask.float()
    n_obs = metric_mask.sum().clamp(min=1.0)

    if recon_logvar is not None:
        if use_student_t_nll:
            df = None
            df_getter = getattr(model, "get_likelihood_df", None) if model is not None else None
            if callable(df_getter):
                df = df_getter(target.shape[-1], device=target.device, dtype=target.dtype)
            nll = _student_t_nll(recon_mean, recon_logvar, target, var_min, var_max, df)
        else:
            logvar_clamped = recon_logvar.clamp(min=np.log(var_min), max=np.log(var_max))
            var = torch.exp(logvar_clamped)
            nll = 0.5 * (logvar_clamped + (target - recon_mean) ** 2 / var)
        recon = _reduce_window_feature_loss(
            nll,
            metric_mask,
            loss_normalization,
            n_chem=n_chem,
            use_family_balanced_loss=use_family_balanced_loss,
            family_loss_chem_weight=family_loss_chem_weight,
            family_loss_scale=family_loss_scale,
            chem_feature_weight=chem_feature_weight,
            psd_feature_weight=psd_feature_weight,
        )
    else:
        recon = _reduce_window_feature_loss(
            (recon_mean - target).square(),
            metric_mask,
            loss_normalization,
            n_chem=n_chem,
            use_family_balanced_loss=use_family_balanced_loss,
            family_loss_chem_weight=family_loss_chem_weight,
            family_loss_scale=family_loss_scale,
            chem_feature_weight=chem_feature_weight,
            psd_feature_weight=psd_feature_weight,
        )

    kl = _latent_kl_loss(mu, logvar, model=model, prior_type=prior_type)
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

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        if config.kl_warmup_epochs is not None:
            warmup_epochs = config.kl_warmup_epochs
        else:
            warmup_ratio = config.kl_warmup_ratio if config.kl_warmup_ratio is not None else 0.2
            warmup_epochs = max(1, int(config.epochs * warmup_ratio))
        self.kl_scheduler = KLAnnealingScheduler(
            config.epochs,
            warmup_epochs,
            max_beta=config.kl_max_beta,
            strategy=config.kl_strategy,
            n_cycles=config.kl_cycles,
            ratio=config.kl_cycle_ratio,
        )
        if config.lr_warmup_epochs is not None:
            lr_warmup_epochs = config.lr_warmup_epochs
        else:
            lr_warmup_ratio = config.lr_warmup_ratio if config.lr_warmup_ratio is not None else 0.05
            lr_warmup_epochs = max(1, int(config.epochs * lr_warmup_ratio))
        self.lr_scheduler = LRWarmupCosineScheduler(
            self.optimizer,
            config.epochs,
            lr_warmup_epochs,
            min_lr=config.lr_min,
        )
        # Matches the 26e reference: linear warmup + cosine runs the whole
        # schedule by default; use_adaptive_lr instead switches to
        # ReduceLROnPlateau (monitoring held-out MSE) once warmup ends.
        self.use_adaptive_lr = bool(config.use_adaptive_lr)
        self.plateau_scheduler = None
        if self.use_adaptive_lr:
            self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.lr_reduce_factor,
                patience=config.lr_reduce_patience,
                threshold=config.lr_reduce_threshold,
                cooldown=config.lr_reduce_cooldown,
                min_lr=config.lr_min,
            )
        self.amp_dtype = self._resolve_amp_dtype(config.amp_dtype)
        self.amp_enabled = bool(config.use_amp and self.amp_dtype is not None)
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

    def _resolve_amp_dtype(self, value):
        if self.device.type != "cuda" or not self.config.use_amp:
            return None
        if value == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[value]

    def _autocast_context(self):
        if not self.amp_enabled or self.amp_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _set_epoch_state(self, epoch):
        # ImputationVAE_Graph forwards this value to the decoder. The research
        # trainer uses it to activate detached variance from epoch 200 onward.
        if hasattr(self.model, "current_epoch"):
            self.model.current_epoch = int(epoch)

    def _run_epoch(self, loader, epoch, train):
        self.model.train(train)
        self._set_epoch_state(epoch)
        beta = self.kl_scheduler.get_beta(epoch)
        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(train):
            # _run_epoch is only ever called with train=True (validation goes
            # through _run_validation instead); per-batch bars are only worth
            # drawing on a live terminal -- see utils.is_interactive.
            for batch in tqdm(loader, desc=f"train epoch {epoch + 1}", leave=False, disable=not is_interactive()):
                input_x = batch["input_x"].to(self.device)
                cond = batch["cond"].to(self.device)
                input_mask = batch["input_mask"].to(self.device)
                target = batch["target"].to(self.device)
                obs_mask = batch["obs_mask"].to(self.device)

                with self._autocast_context():
                    recon_mean, recon_logvar, mu, logvar, _ = self.model(input_x, cond, input_mask)
                    loss, _recon, _kl = vae_loss(
                        recon_mean,
                        recon_logvar,
                        target,
                        obs_mask,
                        mu,
                        logvar,
                        beta,
                        model=self.model,
                        prior_type=self.config.prior_type,
                        use_student_t_nll=self.config.use_student_t_nll,
                        loss_normalization=self.config.loss_normalization,
                        n_chem=int(self.config.model_kwargs.get("n_chem", 0)),
                        use_family_balanced_loss=self.config.use_family_balanced_loss,
                        family_loss_chem_weight=self.config.family_loss_chem_weight,
                        family_loss_scale=self.config.family_loss_scale,
                        chem_feature_weight=self.config.chem_feature_weight,
                        psd_feature_weight=self.config.psd_feature_weight,
                    )

                if train:
                    self.optimizer.zero_grad()
                    if self.grad_scaler.is_enabled():
                        self.grad_scaler.scale(loss).backward()
                        self.grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _run_validation(self, loader, epoch):
        """Evaluate only on the fixed selection-HO positions."""
        self.model.eval()
        self._set_epoch_state(epoch)
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
            for batch in tqdm(loader, desc=f"val epoch {epoch + 1}", leave=False, disable=not is_interactive()):
                input_x = batch["input_x"].to(self.device)
                cond = batch["cond"].to(self.device)
                input_mask = batch["input_mask"].to(self.device)
                target = batch["target"].to(self.device)
                obs_mask = batch["obs_mask"].to(self.device)
                heldout_mask = batch["heldout_mask"].to(self.device)

                with self._autocast_context():
                    recon_mean, recon_logvar, mu, logvar, _ = self.model(input_x, cond, input_mask)
                ho_nll = masked_nll(recon_mean.float(), recon_logvar, target, heldout_mask)
                ho_mse = masked_mse(recon_mean.float(), target, heldout_mask)
                total_nll += ho_nll.item()
                total_mse += ho_mse.item()
                n_batches += 1

                if compute_crps:
                    result = self.model.compute_uncertainty(
                        input_x,
                        cond,
                        input_mask,
                        n_samples=max(2, self.config.val_crps_mc_samples),
                        dist_type=self.config.val_crps_dist_type,
                        return_samples=True,
                        mc_batch_size=self.config.val_mc_batch_size,
                        amp_dtype=self.amp_dtype,
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

        # Outer, whole-run progress bar (epoch X/N, ETA). Only drawn on a
        # live terminal; in a redirected log (nohup, CI) it's a no-op and the
        # per-epoch tqdm.write() lines below are the only output, one clean
        # line per epoch instead of a wall of `\r` fragments.
        epoch_bar = tqdm(
            range(self.config.epochs), desc="training", disable=not is_interactive(), dynamic_ncols=True
        )
        for epoch in epoch_bar:
            in_warmup = epoch < self.lr_scheduler.warmup_epochs
            # Original behavior: cosine scheduler runs every epoch. Adaptive
            # mode: warmup/cosine first, then plateau scheduler post-warmup.
            if not self.use_adaptive_lr or in_warmup:
                self.lr_scheduler.step(epoch)
            train_loss = self._run_epoch(self.train_loader, epoch, train=True)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.val_loader is not None:
                val_metrics = self._run_validation(self.val_loader, epoch)
                metric = val_metrics.get(self.config.validation_metric)
                # CRPS may be intentionally evaluated every N epochs. Do not
                # count skipped epochs against patience.
                if metric is None:
                    metric = None
                tqdm.write(
                    f"epoch {epoch + 1}/{self.config.epochs}  train_loss={train_loss:.4f} "
                    f"val_ho_nll={val_metrics['ho_nll']:.4f} "
                    f"val_ho_mse={val_metrics['ho_mse']:.4f} "
                    f"val_ho_crps={val_metrics['ho_crps'] if val_metrics['ho_crps'] is not None else 'skipped'} "
                    f"lr={current_lr:.2e}"
                )
                epoch_bar.set_postfix(
                    loss=f"{train_loss:.3f}", ho_mse=f"{val_metrics['ho_mse']:.3f}", lr=f"{current_lr:.1e}"
                )
                # Adaptive LR monitors held-out MSE specifically (task-aligned),
                # independent of whichever metric selects the best checkpoint.
                if self.use_adaptive_lr and not in_warmup:
                    self.plateau_scheduler.step(val_metrics["ho_mse"])
            else:
                val_metrics = {"ho_nll": train_loss, "ho_mse": train_loss, "ho_crps": None}
                metric = train_loss
                epoch_bar.set_postfix(loss=f"{train_loss:.3f}", lr=f"{current_lr:.1e}")
                if self.use_adaptive_lr and not in_warmup:
                    self.plateau_scheduler.step(train_loss)

            if metric is not None and metric < best_val - 1e-6:
                best_val = metric
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            elif metric is not None:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    tqdm.write(f"early stopping at epoch {epoch + 1}")
                    break

        epoch_bar.close()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val


def train_from_config(config: TrainConfig, save_path: str) -> float:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    frame = load_frame(config.csv, config.timestamp_col, config.target_cols, config.aux_cols)

    target_raw = frame[config.target_cols].to_numpy(dtype=np.float64)
    target_model_space = transform_target_values(target_raw, config.target_transform)
    aux_raw = (
        frame[config.aux_cols].to_numpy(dtype=np.float64)
        if config.aux_cols
        else np.zeros((len(frame), 0))
    )

    full_data_validation = config.val_fraction == 0.0
    split_idx = chronological_split_index(len(frame), config.val_fraction)
    if full_data_validation:
        train_target = val_target = target_model_space
        train_aux = val_aux = aux_raw
    else:
        train_target, val_target = target_model_space[:split_idx], target_model_space[split_idx:]
        train_aux, val_aux = aux_raw[:split_idx], aux_raw[split_idx:]
    train_aux_mask = ~np.isnan(train_aux)
    val_aux_mask = ~np.isnan(val_aux)

    scaler_target_fit = target_model_space if config.scaler_fit_scope == "full" else train_target
    scaler_aux_fit = aux_raw if config.scaler_fit_scope == "full" else train_aux
    scaler_target = NaNAwareStandardScaler().fit(scaler_target_fit)
    scaler_aux = NaNAwareStandardScaler().fit(scaler_aux_fit) if config.aux_cols else NaNAwareStandardScaler().fit(
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
        "mode": config.dynamic_masking_mode,
        "target_ratio": config.dynamic_mask_target_ratio,
        "mean_duration": config.dynamic_mask_mean_duration,
        "std_duration": config.dynamic_mask_std_duration,
        "min_duration": config.dynamic_mask_min_duration,
        "max_duration": config.dynamic_mask_max_duration,
        "chem_blocks": config.dynamic_mask_chem_blocks,
        "psd_blocks": config.dynamic_mask_psd_blocks,
        "legacy_chem_blocks": 8,
        "legacy_psd_blocks": 6,
        "random_point_drop_prob": config.dynamic_random_point_drop_prob,
        "n_chem": n_chem,
    }
    train_fixed_mask = None
    if len(val_target):
        if config.selection_mask_mode == "anchor_constrained":
            if config.shared_full_heldout_mask:
                selection_full = sample_anchor_constrained_heldout_mask(
                    ~np.isnan(target_model_space),
                    ratio=config.selection_mask_ratio,
                    seed=config.selection_val_seed,
                    n_chem=n_chem,
                )
                if full_data_validation:
                    train_fixed_mask = selection_full
                    val_selection_mask = selection_full
                else:
                    train_fixed_mask = selection_full[:split_idx]
                    val_selection_mask = selection_full[split_idx:]
            else:
                val_selection_mask = sample_anchor_constrained_heldout_mask(
                    ~np.isnan(val_target),
                    ratio=config.selection_mask_ratio,
                    seed=config.selection_val_seed,
                    n_chem=n_chem,
                )
                train_fixed_mask = None
        else:
            val_selection_mask = sample_dynamic_heldout_mask(
                ~np.isnan(val_target),
                {**dynamic_mask_config, "ensure_nonempty": True},
                seed=config.selection_val_seed,
            )
            train_fixed_mask = None
    else:
        val_selection_mask = None
    if len(val_target) and val_selection_mask.sum() == 0:
        raise ValueError("Validation split has no observed target positions for selection-HO validation")

    train_dataset = WindowedTimeSeriesDataset(
        train_target_scaled, train_aux_scaled, config.window_size, config.stride,
        mode="train", denoise_prob=config.denoise_prob, seed=config.seed,
        aux_mask=train_aux_mask, aux_mask_channel=config.aux_mask_channel,
        dynamic_mask_config=dynamic_mask_config, fixed_mask=train_fixed_mask,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training split ({len(train_target)} rows) is shorter than window_size={config.window_size}"
        )

    val_dataset = WindowedTimeSeriesDataset(
        val_target_scaled, val_aux_scaled, config.window_size, config.stride,
        mode="val", seed=config.seed, aux_mask=val_aux_mask,
        aux_mask_channel=config.aux_mask_channel, selection_mask=val_selection_mask,
    )
    if len(val_target) and config.val_fraction > 0 and len(val_dataset) == 0:
        raise ValueError(
            f"Validation split ({len(val_target)} rows) is shorter than window_size={config.window_size}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_loader_num_workers,
        worker_init_fn=_seed_window_worker,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.val_loader_num_workers,
            worker_init_fn=_seed_window_worker,
        )
        if len(val_dataset) > 0
        else None
    )

    model_kwargs = dict(config.model_kwargs)
    model = ImputationVAE_Graph(
        target_dim=len(config.target_cols),
        aux_dim=(2 if config.aux_mask_channel else 1) * len(config.aux_cols),
        window_size=config.window_size,
        **model_kwargs,
    )

    device = setup_device()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[graph-tcn-vae] {len(frame)} rows -> {len(train_dataset)} train / "
        f"{len(val_dataset) if val_loader is not None else 0} val windows "
        f"({len(train_loader)} batches/epoch), {len(config.target_cols)} targets "
        f"({n_chem} chem + {len(config.target_cols) - n_chem} psd), {len(config.aux_cols)} aux cols, "
        f"{n_params:,} params, device={device}, epochs={config.epochs}, batch_size={config.batch_size}"
    )
    trainer = Trainer(model, train_loader, val_loader, config, device)
    best_val = trainer.fit()

    bundle = {
        "bundle_version": 2,
        "state_dict": trainer.model.state_dict(),
        "model_kwargs": model_kwargs,
        "target_cols": list(config.target_cols),
        "aux_cols": list(config.aux_cols),
        "window_size": config.window_size,
        "aux_missing_mode": "mask_channel" if config.aux_mask_channel else "legacy_zero_fill",
        "aux_mask_channel": config.aux_mask_channel,
        "target_transform": config.target_transform,
        "target_output_transform": config.target_output_transform,
        "selection_mask_protocol": f"fixed_{config.selection_mask_mode}_ho",
        "schema": {
            "target_cols": list(config.target_cols),
            "aux_value_cols": list(config.aux_cols),
            "aux_mask_cols": [f"{col}__observed" for col in config.aux_cols],
            "target_dim": len(config.target_cols),
            "cond_dim": (2 if config.aux_mask_channel else 1) * len(config.aux_cols),
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
