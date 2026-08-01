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
import csv
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from .censoring import (
    STATE_CENSORED,
    STATE_MISSING,
    CensoringConfig,
    apply_input_fill,
    build_state_matrix,
    censoring_report,
    high_censoring_columns,
    model_space_thresholds,
)
from .config import TrainConfig
from .contracts import DataSchema
from .data import (
    WindowedTimeSeriesDataset,
    canonicalize_wind_column_names,
    chronological_split_index,
    load_frame,
    load_modality_frame,
    sample_anchor_constrained_heldout_mask,
    sample_block_heldout_mask_to_ratio,
)
from .model_graph_uq import ImputationVAE_Graph
from .preprocessing import (
    fit_auxiliary_scaler,
    fit_target_scaler,
    target_output_transforms,
    transform_auxiliary,
    transform_targets,
)
from .utils import KLAnnealingScheduler, LRWarmupCosineScheduler, is_interactive, setup_device


def _seed_window_worker(_worker_id):
    """Give each DataLoader worker an independent deterministic mask stream."""
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2**32))


def _loader_options(config, num_workers):
    """Build DataLoader options without enabling worker-only knobs at zero."""
    pin_memory = (
        torch.cuda.is_available()
        if config.pin_memory is None
        else bool(config.pin_memory)
    )
    options = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_window_worker,
    }
    if num_workers > 0:
        # Timeline-epoch masks are replaced in the parent dataset before each
        # iterator. Persistent workers would keep stale dataset copies from
        # epoch 0, so they must be respawned for this protocol.
        options["persistent_workers"] = bool(
            config.persistent_workers
            and config.dynamic_mask_scope != "timeline_epoch"
        )
        options["prefetch_factor"] = int(config.loader_prefetch_factor)
    return options


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


def _pointwise_reconstruction_nll(
    recon_mean,
    recon_logvar,
    target,
    *,
    model=None,
    use_student_t_nll=False,
    var_min=1e-3,
    var_max=10.0,
):
    """Return the same pointwise likelihood loss for training and validation."""
    if recon_logvar is None:
        return (recon_mean - target).square()
    if use_student_t_nll:
        df = None
        df_getter = getattr(model, "get_likelihood_df", None) if model is not None else None
        if callable(df_getter):
            df = df_getter(target.shape[-1], device=target.device, dtype=target.dtype)
        return _student_t_nll(
            recon_mean,
            recon_logvar,
            target,
            var_min=var_min,
            var_max=var_max,
            df=df,
        )
    logvar_clamped = recon_logvar.clamp(
        min=np.log(var_min), max=np.log(var_max)
    )
    variance = torch.exp(logvar_clamped)
    return 0.5 * (
        logvar_clamped + (target - recon_mean).square() / variance
    )


_LOG_NDTR_TAIL_CUTOFF = -5.0


def _clamp_signed_denominator(values, floor):
    """Keep continued-fraction denominators away from zero without NaNs."""
    return torch.where(
        values.abs() < floor,
        torch.where(values < 0, values.new_tensor(-floor), values.new_tensor(floor)),
        values,
    )


def _beta_continued_fraction(a, b, x, max_iter=100):
    """Evaluate the regularized incomplete-beta continued fraction."""
    tiny = torch.finfo(x.dtype).tiny
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    d = _clamp_signed_denominator(1.0 - qab * x / qap, tiny)
    d = 1.0 / d
    c = torch.ones_like(d)
    h = d
    for m in range(1, max_iter + 1):
        m_value = float(m)
        m2 = 2.0 * m_value
        aa = m_value * (b - m_value) * x / ((qam + m2) * (a + m2))
        d = _clamp_signed_denominator(1.0 + aa * d, tiny)
        c = _clamp_signed_denominator(1.0 + aa / c, tiny)
        d = 1.0 / d
        h = h * d * c
        aa = -((a + m_value) * (qab + m_value) * x) / (
            (a + m2) * (qap + m2)
        )
        d = _clamp_signed_denominator(1.0 + aa * d, tiny)
        c = _clamp_signed_denominator(1.0 + aa / c, tiny)
        d = 1.0 / d
        h = h * d * c
    return h


def student_t_log_cdf(values, df):
    """Differentiable log-CDF of a standardized Student-t distribution.

    PyTorch exposes Student-t sampling but not ``StudentT.cdf``. The CDF is
    expressed through the regularized incomplete beta function and evaluated
    with its continued fraction, avoiding SciPy in the training graph.
    """
    values = torch.as_tensor(values)
    # The continued fraction is not stable in fp16/bfloat16. Keep the result
    # differentiable while evaluating the special-function path in fp32 under
    # CUDA autocast.
    if values.dtype in {torch.float16, torch.bfloat16}:
        values = values.float()
    df = torch.as_tensor(df, device=values.device, dtype=values.dtype)
    while df.ndim < values.ndim:
        df = df.unsqueeze(0)
    df = df.expand_as(values)
    if torch.any(df <= 0):
        raise ValueError("Student-t degrees of freedom must be positive")

    a = 0.5 * df
    b = torch.full_like(a, 0.5)
    x = (df / (df + values.square())).clamp(min=1e-12, max=1.0)
    reflect = x > (a + 1.0) / (a + b + 2.0)
    x_cf = torch.where(reflect, 1.0 - x, x).clamp(min=1e-12, max=1.0 - 1e-7)
    a_cf = torch.where(reflect, b, a)
    b_cf = torch.where(reflect, a, b)
    log_front = (
        torch.lgamma(a_cf + b_cf)
        - torch.lgamma(a_cf)
        - torch.lgamma(b_cf)
        + a_cf * torch.log(x_cf)
        + b_cf * torch.log1p(-x_cf)
    )
    fraction = _beta_continued_fraction(a_cf, b_cf, x_cf)
    log_small_i = log_front + torch.log(fraction.clamp(min=1e-30)) - torch.log(a_cf)
    small_i = torch.exp(log_small_i).clamp(max=1.0)
    log_i = torch.where(
        reflect,
        torch.log1p(-small_i.clamp(max=1.0 - 1e-7)),
        log_small_i,
    )
    log_half = values.new_tensor(-np.log(2.0))
    return torch.where(
        values < 0,
        log_half + log_i,
        torch.log1p(-0.5 * torch.exp(log_i).clamp(max=1.0)),
    )


def log_ndtr(values):
    """Numerically stable ``log Phi(x)``, identical on every backend.

    ``torch.special.log_ndtr`` is unimplemented on MPS, and dispatching by
    device would make the same config produce different likelihoods on
    different machines. Both branches are always evaluated and clamped, so no
    NaN reaches the backward pass through the unselected side.
    """
    sqrt_half = 0.7071067811865476
    # Right branch: the erfc argument stays below ~3.6, far from underflow.
    upper = torch.log(0.5 * torch.erfc(-values.clamp(min=_LOG_NDTR_TAIL_CUTOFF) * sqrt_half))
    # Left tail: Mills-ratio asymptotic expansion, accurate past |x| >= 5
    # where a direct erfc would underflow to log(0).
    tail = values.clamp(max=_LOG_NDTR_TAIL_CUTOFF)
    tail_sq = tail * tail
    series = 1.0 - 1.0 / tail_sq + 3.0 / tail_sq.square() - 15.0 / (tail_sq * tail_sq.square())
    lower = (
        -0.5 * tail_sq
        - torch.log(-tail)
        - 0.5 * float(np.log(2.0 * np.pi))
        + torch.log(series.clamp(min=1e-12))
    )
    return torch.where(values >= _LOG_NDTR_TAIL_CUTOFF, upper, lower)


def _censored_nll(
    recon_mean,
    recon_logvar,
    threshold,
    var_min=1e-3,
    var_max=10.0,
    *,
    use_student_t_nll=False,
    df=None,
):
    """Tobit term for a left-censored cell: ``-log P(y <= threshold)``.

    ``threshold`` is the detection limit expressed in the scaled space the
    decoder predicts in, broadcast over the feature axis.  Unlike the
    observed-cell likelihood this places probability mass below the limit
    rather than pulling the mean toward a point value.

    Student-t uses the differentiable incomplete-beta CDF above; Gaussian is
    retained for explicit Gaussian likelihoods and old callers.
    """
    if recon_logvar is None:
        # Deterministic decoder: fall back to a hinge that is zero once the
        # prediction is already below the limit.
        return (recon_mean - threshold).clamp(min=0.0).square()
    log_variance = recon_logvar.clamp(min=np.log(var_min), max=np.log(var_max))
    variance = torch.exp(log_variance)
    if use_student_t_nll:
        if df is None:
            df = torch.tensor(3.0, device=recon_mean.device, dtype=recon_mean.dtype)
        df_for_scale = torch.as_tensor(df, device=recon_mean.device, dtype=recon_mean.dtype)
        while df_for_scale.ndim < recon_mean.ndim:
            df_for_scale = df_for_scale.unsqueeze(0)
        variance = (variance * (df_for_scale - 2.0) / df_for_scale).clamp(min=1e-10)
    sigma = torch.sqrt(variance).clamp(min=1e-6)
    standardized_limit = (threshold - recon_mean) / sigma
    if use_student_t_nll:
        return -student_t_log_cdf(standardized_limit, df)
    return -log_ndtr(standardized_limit)


def _likelihood_df(model, num_features, device, dtype):
    getter = getattr(model, "get_likelihood_df", None) if model is not None else None
    if callable(getter):
        return getter(num_features, device=device, dtype=dtype)
    return torch.tensor(3.0, device=device, dtype=dtype)


def _combine_censored_point_loss(
    point_loss,
    metric_mask,
    censor_mask,
    censor_threshold,
    recon_mean,
    recon_logvar,
    var_min,
    var_max,
    use_student_t_nll=False,
    df=None,
):
    """Fold the Tobit term into the pointwise loss and its normalizing mask.

    Observed and censored positions are disjoint, so the two masks can simply
    be added: every supervised cell still contributes exactly once to the
    reduction denominator.
    """
    if censor_mask is None or censor_threshold is None:
        return point_loss, metric_mask
    censor_mask = censor_mask.float()
    if float(censor_mask.sum()) == 0.0:
        return point_loss, metric_mask
    # Uncensored features carry a NaN threshold; their censor mask is zero but
    # a NaN would still propagate through torch.where's backward pass.
    threshold = torch.nan_to_num(censor_threshold, nan=0.0).to(recon_mean.dtype)
    while threshold.ndim < recon_mean.ndim:
        threshold = threshold.unsqueeze(0)
    censored_loss = _censored_nll(
        recon_mean,
        recon_logvar,
        threshold,
        var_min=var_min,
        var_max=var_max,
        use_student_t_nll=use_student_t_nll,
        df=df,
    )
    combined = torch.where(censor_mask.bool(), censored_loss, point_loss)
    return combined, metric_mask + censor_mask


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


def _latent_kl_per_dim(mu, logvar, model=None, prior_type="gaussian"):
    """Per-latent-dimension KL contribution, before summing over latent_dim.

    Returns ``(kl_per_dim, log_det)``. ``log_det`` is the RealNVP flow's
    batch-level Jacobian correction (``None`` outside the flow path) -- it
    mixes dimensions through the coupling layers and isn't decomposable
    per-dimension, so callers doing free-bits clamping must clamp
    ``kl_per_dim`` only and add ``log_det`` back afterward, unclamped.
    """
    mu = torch.clamp(mu, min=-100.0, max=100.0)
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    if model is not None and getattr(model, "last_log_det_J", None) is not None:
        z0 = model.last_z0
        z_k = model.last_zK
        log_det = model.last_log_det_J
        log_q_z0 = -0.5 * (
            np.log(2.0 * np.pi) + logvar + (z0 - mu).square() / logvar.exp()
        )
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
        return log_q_z0 - log_p, log_det
    kl_per_dim = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    return kl_per_dim, None


def _latent_kl_loss(mu, logvar, model=None, prior_type="gaussian"):
    """Raw, unclamped KL divergence -- the posterior-collapse diagnostic
    signal. Always reflects the true divergence, regardless of any
    free-bits floor applied to the loss term in ``vae_loss`` -- otherwise
    collapse would become invisible the moment free-bits is turned on."""
    kl_per_dim, log_det = _latent_kl_per_dim(mu, logvar, model=model, prior_type=prior_type)
    total = kl_per_dim.sum(dim=1)
    if log_det is not None:
        total = total - log_det
    return total.mean()


def vae_loss(recon_mean, recon_logvar, target, obs_mask, mu, logvar, beta, metric_mask=None,
             *, model=None, prior_type="gaussian", use_student_t_nll=False,
             loss_normalization="observed_mean", n_chem=0,
             use_family_balanced_loss=False, family_loss_chem_weight=0.5,
             family_loss_scale="target_dim", chem_feature_weight=1.0,
             psd_feature_weight=1.0, var_min=1e-3, var_max=10.0,
             censor_mask=None, censor_threshold=None, kl_free_bits_nats=0.0):
    obs_mask = obs_mask.float()
    metric_mask = obs_mask if metric_mask is None else metric_mask.float()
    likelihood_df = (
        _likelihood_df(model, target.shape[-1], target.device, target.dtype)
        if use_student_t_nll else None
    )

    if recon_logvar is not None:
        point_nll = _pointwise_reconstruction_nll(
            recon_mean,
            recon_logvar,
            target,
            model=model,
            use_student_t_nll=use_student_t_nll,
            var_min=var_min,
            var_max=var_max,
        )
        point_nll, metric_mask = _combine_censored_point_loss(
            point_nll,
            metric_mask,
            censor_mask,
            censor_threshold,
            recon_mean,
            recon_logvar,
            var_min,
            var_max,
            use_student_t_nll=use_student_t_nll,
            df=likelihood_df,
        )
        recon = _reduce_window_feature_loss(
            point_nll,
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
        point_loss, metric_mask = _combine_censored_point_loss(
            (recon_mean - target).square(),
            metric_mask,
            censor_mask,
            censor_threshold,
            recon_mean,
            recon_logvar,
            var_min,
            var_max,
            use_student_t_nll=use_student_t_nll,
            df=likelihood_df,
        )
        recon = _reduce_window_feature_loss(
            point_loss,
            metric_mask,
            loss_normalization,
            n_chem=n_chem,
            use_family_balanced_loss=use_family_balanced_loss,
            family_loss_chem_weight=family_loss_chem_weight,
            family_loss_scale=family_loss_scale,
            chem_feature_weight=chem_feature_weight,
            psd_feature_weight=psd_feature_weight,
        )

    kl_per_dim, log_det = _latent_kl_per_dim(mu, logvar, model=model, prior_type=prior_type)
    raw_kl_total = kl_per_dim.sum(dim=1)
    if kl_free_bits_nats > 0:
        loss_kl_total = torch.clamp(kl_per_dim, min=kl_free_bits_nats).sum(dim=1)
    else:
        loss_kl_total = raw_kl_total
    if log_det is not None:
        raw_kl_total = raw_kl_total - log_det
        loss_kl_total = loss_kl_total - log_det
    kl = raw_kl_total.mean()
    weighted_kl = beta * loss_kl_total.mean()
    loss = recon + weighted_kl
    return loss, recon.detach(), kl.detach(), weighted_kl.detach()


def empirical_crps_components(samples, target, mask):
    """Return exact empirical CRPS sum without an ``MC x MC`` tensor."""
    mask = mask.float()
    count = mask.sum()
    if count == 0:
        return torch.zeros((), device=target.device), count
    if samples.ndim != target.ndim + 1 or samples.shape[1:] != target.shape:
        raise ValueError("samples must have shape [MC, *target.shape]")
    n_samples = samples.shape[0]
    if n_samples < 2:
        raise ValueError("empirical CRPS requires at least two samples")

    first = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    sorted_samples = torch.sort(samples, dim=0).values
    ranks = torch.arange(
        1,
        n_samples + 1,
        device=samples.device,
        dtype=samples.dtype,
    )
    coefficient_shape = (n_samples,) + (1,) * target.ndim
    coefficients = (
        2.0 * ranks - n_samples - 1.0
    ).reshape(coefficient_shape)
    half_pairwise = (
        sorted_samples * coefficients
    ).sum(dim=0) / float(n_samples * n_samples)
    score = first - half_pairwise
    return (score * mask).sum(), count


def empirical_crps(samples, target, mask):
    """Compute empirical CRPS on selected entries of MC predictive samples."""
    total, count = empirical_crps_components(samples, target, mask)
    if count == 0:
        return torch.tensor(float("nan"), device=target.device)
    return total / count


def masked_mse_components(prediction, target, mask):
    """Return squared-error sum and selected-point count."""
    mask = mask.float()
    return (((prediction - target) ** 2) * mask).sum(), mask.sum()


def masked_mse(prediction, target, mask):
    """Mean squared error over a held-out mask."""
    total, count = masked_mse_components(prediction, target, mask)
    return total / count.clamp(min=1.0)


def masked_nll_components(
    recon_mean,
    recon_logvar,
    target,
    mask,
    *,
    model=None,
    use_student_t_nll=False,
    var_min=1e-3,
    var_max=10.0,
):
    """Return likelihood-consistent NLL sum and selected-point count."""
    point_nll = _pointwise_reconstruction_nll(
        recon_mean,
        recon_logvar,
        target,
        model=model,
        use_student_t_nll=use_student_t_nll,
        var_min=var_min,
        var_max=var_max,
    )
    mask = mask.float()
    return (point_nll * mask).sum(), mask.sum()


def calibration_components(recon_mean, recon_logvar, target, mask,
                           *, var_min=1e-3, var_max=10.0):
    """Split a heteroscedastic NLL into its accuracy and sharpness parts.

    Every heteroscedastic likelihood here has the shape
    ``const + 0.5*log(sigma^2) + f(z^2)`` with ``z = (y - mu) / sigma``. The
    two pieces answer different questions, and a rising NLL cannot say which
    one moved:

    ``z^2``
        Squared standardized residual. Its mean is 1.0 for a well-calibrated
        model; above 1 the predictions are wrong more often than the reported
        variance admits (overconfident), below 1 the variance is inflated.
    ``log sigma^2``
        How sharp the predictive distribution is, independent of whether it is
        centered correctly.

    Returns summed ``z^2``, summed ``log sigma^2``, and the point count.
    """
    zeros = torch.zeros((), device=recon_mean.device)
    if recon_logvar is None:
        return zeros, zeros, zeros
    with torch.no_grad():
        log_variance = recon_logvar.clamp(min=np.log(var_min), max=np.log(var_max))
        variance = torch.exp(log_variance)
        z_squared = (target - recon_mean).square() / variance
        mask = mask.float()
        return (z_squared * mask).sum(), (log_variance * mask).sum(), mask.sum()


def masked_censored_nll_components(
    recon_mean,
    recon_logvar,
    censor_threshold,
    mask,
    *,
    model=None,
    use_student_t_nll=False,
    var_min=1e-3,
    var_max=10.0,
):
    """Tobit NLL sum and count over censored positions.

    Reported alongside the observed-cell held-out NLL so a run that improves
    one at the expense of the other is visible rather than silent.
    """
    mask = mask.float()
    if censor_threshold is None or float(mask.sum()) == 0.0:
        return torch.zeros((), device=recon_mean.device), torch.zeros((), device=recon_mean.device)
    threshold = torch.nan_to_num(censor_threshold, nan=0.0).to(recon_mean.dtype)
    while threshold.ndim < recon_mean.ndim:
        threshold = threshold.unsqueeze(0)
    point_nll = _censored_nll(
        recon_mean,
        recon_logvar,
        threshold,
        var_min=var_min,
        var_max=var_max,
        use_student_t_nll=use_student_t_nll,
        df=(
            _likelihood_df(model, recon_mean.shape[-1], recon_mean.device, recon_mean.dtype)
            if use_student_t_nll else None
        ),
    )
    return (point_nll * mask).sum(), mask.sum()


def masked_nll(
    recon_mean,
    recon_logvar,
    target,
    mask,
    *,
    model=None,
    use_student_t_nll=False,
    var_min=1e-3,
    var_max=10.0,
):
    """Likelihood-consistent reconstruction NLL over a held-out mask."""
    total, count = masked_nll_components(
        recon_mean,
        recon_logvar,
        target,
        mask,
        model=model,
        use_student_t_nll=use_student_t_nll,
        var_min=var_min,
        var_max=var_max,
    )
    return total / count.clamp(min=1.0)


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainConfig, device,
                 censor_threshold=None, history_path=None, train_ho_loader=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_ho_loader = train_ho_loader
        self.config = config
        self.device = device
        # Durable per-epoch record independent of any live terminal or
        # external logging service: just a CSV, so a run can be replotted or
        # compared long after the process exits.
        self.history_path = history_path
        self._history_rows = []
        # Per-feature detection limits in scaled space; None disables the
        # Tobit term entirely and restores the plain observed-only likelihood.
        self.censor_threshold = (
            None
            if censor_threshold is None or not np.isfinite(censor_threshold).any()
            else torch.as_tensor(censor_threshold, dtype=torch.float32, device=device)
        )

        # The 26e reference trainer uses AdamW, not Adam.  Keeping the
        # optimizer choice explicit here matters when weight_decay is part of
        # a reproduction config: Adam and AdamW apply that regularization
        # differently.
        self.optimizer = optim.AdamW(
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
        self.non_blocking_transfer = bool(
            self.device.type == "cuda"
            and (config.pin_memory if config.pin_memory is not None else True)
        )
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

    def _variance_bounds(self):
        decoder = getattr(self.model, "decoder", None)
        return (
            float(getattr(decoder, "var_min", 1e-3)),
            float(getattr(decoder, "var_max", 10.0)),
        )

    def _run_epoch(self, loader, epoch, train, *, beta_override=None):
        self.model.train(train)
        self._set_epoch_state(epoch)
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        beta = (
            self.kl_scheduler.get_beta(epoch)
            if beta_override is None
            else float(beta_override)
        )
        zero = torch.zeros((), device=self.device)
        total_loss = zero.clone()
        n_batches = 0
        total_z2 = zero.clone()
        total_logvar = zero.clone()
        total_points = zero.clone()
        # Raw KL divergence, not beta-weighted: the beta schedule scales how
        # much this term counts toward the loss, but the divergence itself is
        # what tells you whether the posterior is actually collapsing.
        total_kl = zero.clone()
        total_recon = zero.clone()
        total_weighted_kl = zero.clone()

        with torch.set_grad_enabled(train):
            # _run_epoch is only ever called with train=True (validation goes
            # through _run_validation instead); per-batch bars are only worth
            # drawing on a live terminal -- see utils.is_interactive.
            for batch in tqdm(loader, desc=f"train epoch {epoch + 1}", leave=False, disable=not is_interactive()):
                input_x = batch["input_x"].to(self.device, non_blocking=self.non_blocking_transfer)
                cond = batch["cond"].to(self.device, non_blocking=self.non_blocking_transfer)
                input_mask = batch["input_mask"].to(self.device, non_blocking=self.non_blocking_transfer)
                target = batch["target"].to(self.device, non_blocking=self.non_blocking_transfer)
                obs_mask = batch["obs_mask"].to(self.device, non_blocking=self.non_blocking_transfer)
                censor_mask = (
                    batch["censor_mask"].to(self.device, non_blocking=self.non_blocking_transfer)
                    if self.censor_threshold is not None and "censor_mask" in batch
                    else None
                )

                with self._autocast_context():
                    recon_mean, recon_logvar, mu, logvar, _ = self.model(input_x, cond, input_mask)
                    var_min, var_max = self._variance_bounds()
                    loss, _recon, _kl, _weighted_kl = vae_loss(
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
                        var_min=var_min,
                        var_max=var_max,
                        censor_mask=censor_mask,
                        censor_threshold=self.censor_threshold,
                        kl_free_bits_nats=self.config.kl_free_bits_nats,
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

                # Same cells the reconstruction term is scored on, so the
                # breakdown is directly comparable to the held-out one.
                z2_sum, logvar_sum, count = calibration_components(
                    recon_mean.float(), recon_logvar, target, obs_mask,
                    var_min=var_min, var_max=var_max,
                )
                total_z2 += z2_sum.detach()
                total_logvar += logvar_sum.detach()
                total_points += count.detach()

                total_loss += loss.detach()
                total_kl += _kl.detach()
                total_recon += _recon.detach()
                total_weighted_kl += _weighted_kl.detach()
                n_batches += 1

        points = max(total_points, 1.0)
        return float((total_loss / max(n_batches, 1)).item()), {
            "z2": float((total_z2 / points).item()),
            "log_sigma2": float((total_logvar / points).item()),
            "kl": float((total_kl / max(n_batches, 1)).item()),
            "recon": float((total_recon / max(n_batches, 1)).item()),
            "weighted_kl": float((total_weighted_kl / max(n_batches, 1)).item()),
        }

    def _run_validation(self, loader, epoch, loader_name="Validation"):
        """Evaluate only on the fixed selection-HO positions."""
        self.model.eval()
        self._set_epoch_state(epoch)
        zero = torch.zeros((), device=self.device)
        total_nll = zero.clone()
        total_mse = zero.clone()
        total_heldout_count = zero.clone()
        total_crps = zero.clone()
        total_crps_count = zero.clone()
        total_censored_nll = zero.clone()
        total_censored_count = zero.clone()
        total_z2 = zero.clone()
        total_logvar = zero.clone()
        total_kl = zero.clone()
        n_batches = 0
        compute_crps = (
            self.config.validation_metric == "ho_crps"
            and (epoch == 0 or (epoch + 1) % max(1, self.config.val_crps_every_n_epochs) == 0)
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"val epoch {epoch + 1}", leave=False, disable=not is_interactive()):
                input_x = batch["input_x"].to(self.device, non_blocking=self.non_blocking_transfer)
                cond = batch["cond"].to(self.device, non_blocking=self.non_blocking_transfer)
                input_mask = batch["input_mask"].to(self.device, non_blocking=self.non_blocking_transfer)
                target = batch["target"].to(self.device, non_blocking=self.non_blocking_transfer)
                obs_mask = batch["obs_mask"].to(self.device, non_blocking=self.non_blocking_transfer)
                heldout_mask = batch["heldout_mask"].to(self.device, non_blocking=self.non_blocking_transfer)

                with self._autocast_context():
                    recon_mean, recon_logvar, mu, logvar, _ = self.model(input_x, cond, input_mask)
                total_kl += _latent_kl_loss(
                    mu, logvar, model=self.model, prior_type=self.config.prior_type
                ).detach()
                n_batches += 1
                var_min, var_max = self._variance_bounds()
                nll_sum, heldout_count = masked_nll_components(
                    recon_mean.float(),
                    recon_logvar,
                    target,
                    heldout_mask,
                    model=self.model,
                    use_student_t_nll=self.config.use_student_t_nll,
                    var_min=var_min,
                    var_max=var_max,
                )
                mse_sum, mse_count = masked_mse_components(
                    recon_mean.float(), target, heldout_mask
                )
                z2_sum, logvar_sum, _ = calibration_components(
                    recon_mean.float(), recon_logvar, target, heldout_mask,
                    var_min=var_min, var_max=var_max,
                )
                total_z2 += z2_sum.detach()
                total_logvar += logvar_sum.detach()

                total_nll += nll_sum.detach()
                total_mse += mse_sum.detach()
                total_heldout_count += heldout_count.detach()

                if self.censor_threshold is not None and "censor_mask" in batch:
                    cens_sum, cens_count = masked_censored_nll_components(
                        recon_mean.float(),
                        recon_logvar,
                        self.censor_threshold,
                        batch["censor_mask"].to(self.device, non_blocking=self.non_blocking_transfer),
                        model=self.model,
                        use_student_t_nll=self.config.use_student_t_nll,
                        var_min=var_min,
                        var_max=var_max,
                    )
                    total_censored_nll += cens_sum.detach()
                    total_censored_count += cens_count.detach()

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
                    crps_sum, crps_count = empirical_crps_components(
                        samples, target, heldout_mask
                    )
                    valid_crps = (crps_count > 0) & torch.isfinite(crps_sum)
                    total_crps += torch.where(
                        valid_crps, crps_sum, torch.zeros_like(crps_sum)
                    ).detach()
                    total_crps_count += torch.where(
                        valid_crps, crps_count, torch.zeros_like(crps_count)
                    ).detach()

        heldout_count_value = float(total_heldout_count.item())
        censored_count_value = float(total_censored_count.item())
        crps_count_value = float(total_crps_count.item())
        if heldout_count_value <= 0:
            raise ValueError(f"{loader_name} loader contained no held-out target points")
        return {
            "ho_nll": float((total_nll / heldout_count_value).item()),
            "ho_mse": float((total_mse / heldout_count_value).item()),
            "ho_crps": (
                float((total_crps / crps_count_value).item())
                if crps_count_value > 0
                else None
            ),
            # Reported, not selected on: model selection stays on the
            # observed held-out set so the metric keeps a ground-truth scalar.
            "censored_nll": (
                float((total_censored_nll / censored_count_value).item())
                if censored_count_value > 0
                else None
            ),
            # Mean squared standardized residual over the held-out cells:
            # 1.0 is calibrated, >1 overconfident, <1 over-dispersed.
            "z2": float((total_z2 / heldout_count_value).item()),
            "log_sigma2": float((total_logvar / heldout_count_value).item()),
            # Raw KL divergence, batch-averaged like train_loss (not
            # cell-weighted: it's one scalar per window, not per feature-cell).
            "kl": float((total_kl / max(n_batches, 1)).item()),
        }

    _HISTORY_FIELDS = [
        "epoch", "train_loss", "val_ho_nll", "val_ho_mse", "val_ho_crps",
        "val_censored_nll", "train_ho_nll", "train_ho_mse", "train_ho_crps",
        "train_ho_z2", "train_ho_log_sigma2",
        "train_z2", "train_log_sigma2", "val_ho_z2", "val_ho_log_sigma2",
        # Raw KL divergence (not beta-weighted): the direct posterior-collapse
        # diagnostic. train_kl is over directly-supervised cells each batch;
        # val_kl/train_ho_kl are over the respective held-out windows.
        "train_kl", "val_kl", "train_ho_kl",
        # Loss decomposition: train_loss == train_recon + train_weighted_kl
        # (train_weighted_kl = kl_beta * the free-bits-floored KL used for
        # the gradient -- can exceed kl_beta * train_kl once free-bits is
        # clamping some dimensions up, since train_kl above stays raw).
        "train_recon", "train_weighted_kl",
        "lr", "kl_beta", "is_best",
    ]

    def _record_epoch(self, epoch, train_loss, val_metrics, current_lr, beta, is_best,
                      train_calibration=None, train_ho_metrics=None):
        if self.history_path is None:
            return
        train_calibration = train_calibration or {}
        train_ho_metrics = train_ho_metrics or {}
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_ho_nll": val_metrics.get("ho_nll"),
            "val_ho_mse": val_metrics.get("ho_mse"),
            "val_ho_crps": val_metrics.get("ho_crps"),
            "val_censored_nll": val_metrics.get("censored_nll"),
            "train_ho_nll": train_ho_metrics.get("ho_nll"),
            "train_ho_mse": train_ho_metrics.get("ho_mse"),
            "train_ho_crps": train_ho_metrics.get("ho_crps"),
            "train_ho_z2": train_ho_metrics.get("z2"),
            "train_ho_log_sigma2": train_ho_metrics.get("log_sigma2"),
            # Accuracy/sharpness split of the same NLL, so a train-vs-held-out
            # gap can be attributed rather than just observed.
            "train_z2": train_calibration.get("z2"),
            "train_log_sigma2": train_calibration.get("log_sigma2"),
            "val_ho_z2": val_metrics.get("z2"),
            "val_ho_log_sigma2": val_metrics.get("log_sigma2"),
            "train_kl": train_calibration.get("kl"),
            "val_kl": val_metrics.get("kl"),
            "train_ho_kl": train_ho_metrics.get("kl"),
            "train_recon": train_calibration.get("recon"),
            "train_weighted_kl": train_calibration.get("weighted_kl"),
            "lr": current_lr,
            "kl_beta": beta,
            "is_best": int(is_best),
        }
        self._history_rows.append(row)
        # Rewritten whole each epoch rather than appended: cheap at these row
        # counts, and it means a killed run's file is always valid CSV with a
        # header, never a half-written line missing one.
        os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
        with open(self.history_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(self._history_rows)

    def fit(self):
        best_val = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.best_epoch = None
        self.epochs_completed = 0

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
            train_loss, train_calibration = self._run_epoch(self.train_loader, epoch, train=True)
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_ho_metrics = None
            if self.train_ho_loader is not None:
                train_ho_metrics = self._run_validation(
                    self.train_ho_loader, epoch, loader_name="Train-HO"
                )

            if self.val_loader is not None:
                val_metrics = self._run_validation(self.val_loader, epoch, loader_name="Validation")
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
                    + (
                        f"val_cens_nll={val_metrics['censored_nll']:.4f} "
                        if val_metrics.get("censored_nll") is not None else ""
                    )
                    + (
                        f"train_ho_nll={train_ho_metrics['ho_nll']:.4f} "
                        f"train_ho_mse={train_ho_metrics['ho_mse']:.4f} "
                        if train_ho_metrics is not None else ""
                    )
                    + f"z2={train_calibration['z2']:.2f}/{val_metrics['z2']:.2f} "
                    + f"kl={train_calibration['kl']:.3f}/{val_metrics['kl']:.3f} "
                    + f"lr={current_lr:.2e}"
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

            is_best = metric is not None and metric < best_val - 1e-6
            if is_best:
                best_val = metric
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                epochs_without_improvement = 0
            elif metric is not None:
                epochs_without_improvement += 1
            self._record_epoch(
                epoch, train_loss, val_metrics, current_lr,
                self.kl_scheduler.get_beta(epoch), is_best,
                train_calibration=train_calibration,
                train_ho_metrics=train_ho_metrics,
            )
            self.epochs_completed = epoch + 1
            if metric is not None and not is_best and epochs_without_improvement >= self.config.patience:
                tqdm.write(f"early stopping at epoch {epoch + 1}")
                break

        epoch_bar.close()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_val

    def refit_full_data(self, train_loader, monitor_loader, history_path=None):
        """Continue from the selected checkpoint after restoring global-HO cells.

        This is a final-fit phase, not an independent validation phase. The
        monitor loader applies one deterministic mask shaped like the dynamic
        training masks so its MSE is stable enough for an optional practical
        stopping rule. Those target values remain part of full-data training,
        so the metric must not be reported as generalization performance.
        """
        max_epochs = int(self.config.full_data_refit_epochs)
        if max_epochs <= 0:
            return None

        refit_lr = (
            float(self.config.full_data_refit_lr)
            if self.config.full_data_refit_lr is not None
            else max(float(self.config.lr_min), float(self.config.lr) * 0.1)
        )
        patience = self.config.full_data_refit_patience
        early_stopping_enabled = patience is not None

        # The selected model weights carry over, but optimizer and schedule
        # state do not. A small constant LR and the fully annealed KL weight
        # make this a continuation phase rather than a second warmup.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=refit_lr,
            weight_decay=self.config.weight_decay,
        )
        self.train_loader = train_loader
        best_monitor_mse = float("inf")
        best_state = None
        best_epoch = None
        epochs_without_improvement = 0
        rows = []
        base_model_epoch = (
            self.best_epoch + 1
            if self.best_epoch is not None
            else self.epochs_completed
        )

        epoch_bar = tqdm(
            range(max_epochs),
            desc="full-data refit",
            disable=not is_interactive(),
            dynamic_ncols=True,
        )
        for phase_epoch in epoch_bar:
            model_epoch = base_model_epoch + phase_epoch
            train_loss, train_calibration = self._run_epoch(
                train_loader,
                model_epoch,
                train=True,
                beta_override=self.config.kl_max_beta,
            )
            monitor_metrics = self._run_validation(
                monitor_loader,
                model_epoch,
                loader_name="Refit dynamic-HO monitor",
            )
            monitor_mse = float(monitor_metrics["ho_mse"])
            is_best = monitor_mse < best_monitor_mse - 1e-6
            if is_best:
                best_monitor_mse = monitor_mse
                best_epoch = phase_epoch
                epochs_without_improvement = 0
                if early_stopping_enabled:
                    best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1

            rows.append({
                "epoch": phase_epoch + 1,
                "model_epoch": model_epoch + 1,
                "train_loss": train_loss,
                "dynamic_ho_nll": monitor_metrics["ho_nll"],
                "dynamic_ho_mse": monitor_mse,
                "dynamic_ho_z2": monitor_metrics["z2"],
                "dynamic_ho_log_sigma2": monitor_metrics["log_sigma2"],
                "train_z2": train_calibration["z2"],
                "train_log_sigma2": train_calibration["log_sigma2"],
                # Beta is pinned to kl_max_beta for the whole refit phase, but
                # the divergence itself isn't -- this is what actually shows
                # whether the posterior keeps collapsing under the low, fixed
                # refit LR, which a constant kl_beta column can't tell you.
                "train_kl": train_calibration["kl"],
                "dynamic_ho_kl": monitor_metrics["kl"],
                "train_recon": train_calibration["recon"],
                "train_weighted_kl": train_calibration["weighted_kl"],
                "lr": refit_lr,
                "kl_beta": self.config.kl_max_beta,
                "is_best": int(is_best),
            })
            if history_path is not None:
                os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
                with open(history_path, "w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            tqdm.write(
                f"refit {phase_epoch + 1}/{max_epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"dynamic_ho_mse={monitor_mse:.4f}  "
                f"lr={refit_lr:.2e}"
            )
            epoch_bar.set_postfix(
                loss=f"{train_loss:.3f}",
                dynamic_ho_mse=f"{monitor_mse:.3f}",
            )
            if (
                early_stopping_enabled
                and not is_best
                and epochs_without_improvement >= patience
            ):
                tqdm.write(
                    "full-data refit early stopping at "
                    f"epoch {phase_epoch + 1} on dynamic_ho_mse"
                )
                break

        epoch_bar.close()
        if early_stopping_enabled and best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "epochs_completed": len(rows),
            "max_epochs": max_epochs,
            "early_stopping_enabled": early_stopping_enabled,
            "patience": patience,
            "learning_rate": refit_lr,
            "monitor_metric": "dynamic_ho_mse",
            "best_monitor_mse": best_monitor_mse,
            "best_epoch": None if best_epoch is None else best_epoch + 1,
            "history_csv": (
                None if history_path is None else os.path.basename(history_path)
            ),
        }


def _make_fixed_ho_mask(observed_mask, *, mode, ratio, seed, dynamic_config, n_chem):
    """Build a deterministic HO mask without exposing its targets."""
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if mode == "anchor_constrained":
        return sample_anchor_constrained_heldout_mask(
            observed_mask, ratio=ratio, seed=seed, n_chem=n_chem
        )
    return sample_block_heldout_mask_to_ratio(
        observed_mask,
        {**dynamic_config, "target_ratio": ratio, "ensure_nonempty": True},
        seed=seed,
    )


def load_external_heldout_mask(
    mask_path,
    columns_path,
    *,
    expected_rows,
    target_cols,
    observed_mask,
):
    """Load and validate an externally generated full-timeline HO mask.

    The matrix is deliberately not intersected with ``observed_mask`` here:
    the supplied mask is a benchmark artifact and must remain byte-for-byte
    semantically intact. ``WindowedTimeSeriesDataset`` already intersects a
    fixed mask with the actually observed cells when constructing its
    training/evaluation masks. The overlap is reported so a changed QC
    source cannot silently look like the original benchmark.
    """
    if not mask_path:
        raise ValueError("mask_path is required")
    raw = np.load(mask_path, allow_pickle=False)
    if raw.ndim != 2:
        raise ValueError(
            f"External held-out mask must be 2-D, got shape {raw.shape}"
        )
    expected_shape = (int(expected_rows), len(target_cols))
    if tuple(raw.shape) != expected_shape:
        raise ValueError(
            "External held-out mask shape does not match the loaded data: "
            f"expected={expected_shape}, got={tuple(raw.shape)}"
        )
    if raw.dtype == np.bool_:
        mask = raw.copy()
    else:
        if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
            raise ValueError("External held-out mask must contain only boolean/0/1 values")
        mask = raw.astype(bool)

    target_cols = [str(column) for column in target_cols]
    if columns_path:
        columns_frame = pd.read_csv(columns_path)
        if list(columns_frame.columns) != ["target_col"]:
            raise ValueError(
                "External held-out mask columns file must contain exactly one "
                "'target_col' column"
            )
        mask_cols = columns_frame["target_col"].astype(str).tolist()
        if mask_cols != target_cols:
            raise ValueError(
                "External held-out mask columns do not match the loaded target order: "
                f"mask_first={mask_cols[:5]}, target_first={target_cols[:5]}"
            )

    observed_mask = np.asarray(observed_mask, dtype=bool)
    if observed_mask.shape != mask.shape:
        raise ValueError(
            "observed_mask shape does not match external held-out mask: "
            f"observed={observed_mask.shape}, mask={mask.shape}"
        )
    overlap = mask & ~observed_mask
    diagnostics = {
        "source": os.fspath(mask_path),
        "columns_source": None if columns_path is None else os.fspath(columns_path),
        "shape": [int(value) for value in mask.shape],
        "requested_cells": int(mask.sum()),
        "observed_cells": int((mask & observed_mask).sum()),
        "natural_missing_overlap_cells": int(overlap.sum()),
    }
    return mask, diagnostics


# Kept as a compatibility alias for the initial private helper name. New
# callers should use the public loader so training and held-out evaluation can
# share exactly the same validation and provenance behavior.
_load_external_heldout_mask = load_external_heldout_mask


def _build_bundle(
    *, trainer, config, model_kwargs, target_cols, aux_cols, preprocessing,
    censoring, censor_threshold_scaled, censor_report, history_path,
    refit_history_path, refit_summary, selection_full, best_val,
    selection_mask_diagnostics,
    data_schema, data_interface, scaler_target, scaler_aux,
):
    """Assemble a loadable checkpoint bundle from the trainer's current weights.

    Factored out so a stage-1 (pre-refit) checkpoint and the final (post-refit)
    checkpoint can both be built from the same logic -- the only difference
    between the two calls is which state_dict/refit fields are passed in.
    """
    input_transforms = {
        "chemistry": preprocessing.chemistry.transform,
        "psd": preprocessing.psd.transform,
        "meteorology": preprocessing.meteorology.transform,
    }
    output_transforms = target_output_transforms(data_schema, preprocessing)
    present_input_transforms = []
    if data_schema.chemistry_cols:
        present_input_transforms.append(input_transforms["chemistry"])
    if data_schema.psd_cols:
        present_input_transforms.append(input_transforms["psd"])
    uniform_input_transform = (
        present_input_transforms[0]
        if len(set(present_input_transforms)) == 1
        else "mixed"
    )
    uniform_output_transform = (
        output_transforms[0]
        if output_transforms and len(set(output_transforms)) == 1
        else "mixed"
    )
    return {
        "bundle_version": 3,
        "architecture_version": 1,
        "state_dict_format_version": 1,
        "state_dict": trainer.model.state_dict(),
        "model_kwargs": model_kwargs,
        "target_cols": list(target_cols),
        "aux_cols": list(aux_cols),
        "window_size": config.window_size,
        "stride": config.stride,
        "aux_missing_mode": (
            "mask_channel" if preprocessing.aux_mask_channel else "legacy_zero_fill"
        ),
        "aux_mask_channel": preprocessing.aux_mask_channel,
        "target_transform": uniform_input_transform,
        "target_output_transform": uniform_output_transform,
        "target_output_transforms": output_transforms,
        "preprocessing": preprocessing.to_dict(),
        "censoring": censoring.to_dict(),
        # Scaled-space limits so inference can flag non-detects and report
        # P(y <= MDL) without re-deriving the transform chain.
        "censor_threshold_scaled": [
            None if not np.isfinite(value) else float(value)
            for value in censor_threshold_scaled
        ],
        "censoring_report": censor_report,
        "history_csv": os.path.basename(history_path),
        "refit_history_csv": (
            None
            if refit_history_path is None
            else os.path.basename(refit_history_path)
        ),
        "training_summary": {
            "selection_metric": config.validation_metric,
            "selection_best_value": float(best_val),
            "selection_best_epoch": (
                None if trainer.best_epoch is None else trainer.best_epoch + 1
            ),
            "selection_epochs_completed": trainer.epochs_completed,
            "global_heldout_cells": (
                None if selection_full is None else int(selection_full.sum())
            ),
            "global_heldout_seed": (
                config.selection_val_seed
                if selection_full is not None
                else None
            ),
            "external_selection_mask": selection_mask_diagnostics,
            "refit": refit_summary,
        },
        "data_schema": data_schema.to_dict(),
        "data_interface": data_interface,
        "time_grid": {
            "frequency": data_schema.frequency,
            "timezone": data_schema.timezone,
            "policy": data_schema.time_grid_policy,
            "duplicate_timestamp_policy": data_schema.duplicate_timestamp_policy,
        },
        "selection_mask_protocol": (
            f"shared_full_fixed_{config.selection_mask_mode}_ho"
            if selection_full is not None
            else f"fixed_{config.selection_mask_mode}_ho"
        ),
        "schema": {
            "target_cols": list(target_cols),
            "aux_value_cols": list(aux_cols),
            "aux_mask_cols": [f"{col}__observed" for col in aux_cols],
            "target_dim": data_schema.target_dim,
            "cond_dim": (
                (2 if preprocessing.aux_mask_channel else 1) * data_schema.aux_dim
            ),
        },
        "scaler_target": scaler_target.to_dict(),
        "scaler_aux": scaler_aux.to_dict(),
        "config": config.to_dict(),
    }


def train_from_config(
    config: TrainConfig,
    save_path: str,
    *,
    prepared_data=None,
) -> float:
    """Train from a config and optionally preloaded ``(frame, DataSchema)``.

    ``prepared_data`` is the internal seam used by the high-level DataFrame
    interface. It keeps runtime data out of the serialized training config.
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model_kwargs = dict(config.model_kwargs)
    if prepared_data is not None:
        if not isinstance(prepared_data, tuple) or len(prepared_data) != 2:
            raise TypeError("prepared_data must be a (DataFrame, DataSchema) tuple")
        frame, data_schema = prepared_data
        if isinstance(data_schema, dict):
            data_schema = DataSchema.from_dict(data_schema)
        if not isinstance(frame, pd.DataFrame) or not isinstance(data_schema, DataSchema):
            raise TypeError("prepared_data must contain a pandas DataFrame and DataSchema")
        missing_columns = [
            column
            for column in [*data_schema.target_cols, *data_schema.auxiliary_cols]
            if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(f"Prepared frame is missing schema columns: {missing_columns}")
        frame = frame[[*data_schema.target_cols, *data_schema.auxiliary_cols]].copy()
        configured_n_chem = model_kwargs.get("n_chem")
        if configured_n_chem is not None and int(configured_n_chem) != data_schema.n_chem:
            raise ValueError(
                f"model n_chem={configured_n_chem} conflicts with prepared chemistry "
                f"columns ({data_schema.n_chem})"
            )
        model_kwargs["n_chem"] = data_schema.n_chem
        data_interface = "in_memory_modalities"
    elif config.modality_files is not None:
        frame, data_schema = load_modality_frame(
            config.modality_files,
            config.timestamp_col,
            expected_frequency=config.expected_frequency,
            time_grid_policy=config.time_grid_policy,
            duplicate_timestamp_policy=config.duplicate_timestamp_policy,
            add_time_cyclical_features=config.add_time_cyclical_features,
        )
        configured_n_chem = model_kwargs.get("n_chem")
        if configured_n_chem is not None and int(configured_n_chem) != data_schema.n_chem:
            raise ValueError(
                f"model n_chem={configured_n_chem} conflicts with discovered chemistry "
                f"columns ({data_schema.n_chem})"
            )
        model_kwargs["n_chem"] = data_schema.n_chem
        data_interface = "modality_files"
    elif config.csv:
        frame = load_frame(
            config.csv,
            config.timestamp_col,
            config.target_cols,
            config.aux_cols,
            expected_frequency=config.expected_frequency,
            time_grid_policy=config.time_grid_policy,
            duplicate_timestamp_policy=config.duplicate_timestamp_policy,
            canonicalize_wind=True,
        )
        resolved_aux_cols = canonicalize_wind_column_names(config.aux_cols)
        n_chem = int(model_kwargs.get("n_chem", 0))
        if not 0 <= n_chem <= len(config.target_cols):
            raise ValueError("n_chem must be between 0 and the number of target columns")
        data_schema = DataSchema(
            timestamp_col=config.timestamp_col,
            chemistry_cols=list(config.target_cols[:n_chem]),
            psd_cols=list(config.target_cols[n_chem:]),
            meteorology_cols=resolved_aux_cols,
            frequency=frame.attrs.get("frequency"),
            timezone=frame.attrs.get("timezone"),
            time_grid_policy=config.time_grid_policy,
            duplicate_timestamp_policy=config.duplicate_timestamp_policy,
        )
        model_kwargs["n_chem"] = n_chem
        data_interface = "legacy_columns"
    else:
        raise ValueError(
            "TrainConfig contains no data source; provide modality_files, legacy CSV fields, "
            "or prepared_data"
        )

    preprocessing = config.preprocessing
    target_cols = data_schema.target_cols
    aux_cols = data_schema.auxiliary_cols
    n_chem = data_schema.n_chem
    target_raw = frame[target_cols].to_numpy(dtype=np.float64)
    censoring = config.censoring or CensoringConfig()
    # Classify before transforming: detection is defined on physical values.
    state_full = build_state_matrix(target_raw, data_schema, censoring)
    censor_full = state_full == STATE_CENSORED
    # loss='ignore' demotes non-detects to missing, so drop their values too.
    if censoring.active and censoring.loss == "ignore":
        target_raw = np.where(state_full == STATE_MISSING, np.nan, target_raw)
    target_input_raw = apply_input_fill(target_raw, state_full, data_schema, censoring)
    target_model_space = transform_targets(target_input_raw, data_schema, preprocessing)
    aux_raw = (
        frame[aux_cols].to_numpy(dtype=np.float64)
        if aux_cols
        else np.zeros((len(frame), 0), dtype=np.float64)
    )
    aux_model_space = transform_auxiliary(aux_raw, preprocessing)

    full_data_validation = config.val_fraction == 0.0
    split_idx = chronological_split_index(len(frame), config.val_fraction)
    if full_data_validation:
        train_target = val_target = target_model_space
        train_aux = val_aux = aux_model_space
        train_censor = val_censor = censor_full
    else:
        train_target, val_target = target_model_space[:split_idx], target_model_space[split_idx:]
        train_aux, val_aux = aux_model_space[:split_idx], aux_model_space[split_idx:]
        train_censor, val_censor = censor_full[:split_idx], censor_full[split_idx:]
    train_aux_mask = ~np.isnan(train_aux)
    val_aux_mask = ~np.isnan(val_aux)

    scaler_target_fit = target_model_space if preprocessing.fit_scope == "full" else train_target
    scaler_aux_fit = aux_model_space if preprocessing.fit_scope == "full" else train_aux
    # Fitting on substituted non-detects would describe the detection-limit
    # spike rather than the concentration distribution, dragging the center
    # down and the spread with it. Hide them from the fit only.
    scaler_censor_fit = censor_full if preprocessing.fit_scope == "full" else train_censor
    if scaler_censor_fit.any():
        hidden = np.where(scaler_censor_fit, np.nan, scaler_target_fit)
        # A column that is censored almost everywhere has too few detects left
        # to fit an affine scaler. Keep its non-detects in the fit rather than
        # failing the run, and name it: such a column carries little signal and
        # is usually better dropped from the target set entirely.
        too_sparse = np.flatnonzero((~np.isnan(hidden)).sum(axis=0) < 2)
        if len(too_sparse):
            hidden[:, too_sparse] = scaler_target_fit[:, too_sparse]
            print(
                "[graph-temporal-vae] scaler fit kept non-detects for "
                f"{len(too_sparse)} near-fully-censored column(s): "
                f"{[target_cols[i] for i in too_sparse]}"
            )
        scaler_target_fit = hidden
    scaler_target = fit_target_scaler(scaler_target_fit, data_schema, preprocessing)
    scaler_aux = fit_auxiliary_scaler(scaler_aux_fit, preprocessing)

    train_target_scaled = scaler_target.transform(train_target)
    train_target_scaled[np.isnan(train_target)] = np.nan
    val_target_scaled = scaler_target.transform(val_target)
    val_target_scaled[np.isnan(val_target)] = np.nan
    # Detection limits must live in the same space the decoder predicts in.
    censor_threshold_scaled = model_space_thresholds(
        data_schema, preprocessing, scaler_target, censoring
    )
    censor_report = censoring_report(state_full, data_schema, censoring)
    train_aux_scaled = scaler_aux.transform(train_aux) if aux_cols else train_aux
    val_aux_scaled = scaler_aux.transform(val_aux) if aux_cols else val_aux

    dynamic_mask_config = {
        "mode": config.dynamic_masking_mode,
        "scope": config.dynamic_mask_scope,
        "target_ratio": config.dynamic_mask_target_ratio,
        "mean_duration": config.dynamic_mask_mean_duration,
        "std_duration": config.dynamic_mask_std_duration,
        "min_duration": config.dynamic_mask_min_duration,
        "max_duration": config.dynamic_mask_max_duration,
        "duration_source": config.dynamic_mask_duration_source,
        "chem_blocks": config.dynamic_mask_chem_blocks,
        "psd_blocks": config.dynamic_mask_psd_blocks,
        "legacy_chem_blocks": 8,
        "legacy_psd_blocks": 6,
        "random_point_drop_prob": config.dynamic_random_point_drop_prob,
        "n_chem": n_chem,
    }
    train_fixed_mask = None
    train_selection_mask = None
    selection_full = None
    selection_mask_diagnostics = None
    val_selection_mask = None
    # Held-out selection scores predictions against a ground-truth scalar, so
    # only real detections are eligible: a non-detect has no such value.
    observed_full = (~np.isnan(target_model_space)) & ~censor_full
    observed_train = (~np.isnan(train_target)) & ~train_censor
    observed_val = (~np.isnan(val_target)) & ~val_censor

    if full_data_validation and config.shared_full_heldout_mask:
        # One global fixed mask serves both roles: it is excluded from the
        # stage-one input/loss and scored by the validation loader. This is the
        # research-repo protocol and works for both block and anchor masks.
        if config.selection_mask_path:
            selection_full, selection_mask_diagnostics = _load_external_heldout_mask(
                config.selection_mask_path,
                config.selection_mask_columns_path,
                expected_rows=len(frame),
                target_cols=target_cols,
                observed_mask=observed_full,
            )
            if selection_mask_diagnostics["natural_missing_overlap_cells"]:
                print(
                    "[graph-temporal-vae] external HO mask: "
                    f"{selection_mask_diagnostics['natural_missing_overlap_cells']} "
                    "requested cells overlap natural missingness; they will not "
                    "contribute to HO metrics"
                )
        else:
            selection_full = _make_fixed_ho_mask(
                observed_full,
                mode=config.selection_mask_mode,
                ratio=config.selection_mask_ratio,
                seed=config.selection_val_seed,
                dynamic_config=dynamic_mask_config,
                n_chem=n_chem,
            )
        train_fixed_mask = selection_full
        val_selection_mask = selection_full
    elif config.train_ho_enabled:
        train_selection_mask = _make_fixed_ho_mask(
            observed_train,
            mode=config.selection_mask_mode,
            ratio=config.train_ho_ratio,
            seed=config.train_ho_seed,
            dynamic_config=dynamic_mask_config,
            n_chem=n_chem,
        )
        if train_selection_mask.sum() == 0:
            raise ValueError("Training split has no observed target positions for train-HO evaluation")
        # WindowedTimeSeriesDataset removes fixed_mask cells from obs_mask, so
        # these values are hidden from both the input and the training loss.
        train_fixed_mask = train_selection_mask

    if val_selection_mask is None and len(val_target):
        val_selection_mask = _make_fixed_ho_mask(
            observed_val,
            mode=config.selection_mask_mode,
            ratio=config.selection_mask_ratio,
            seed=config.selection_val_seed,
            dynamic_config=dynamic_mask_config,
            n_chem=n_chem,
        )
    if val_selection_mask is None:
        if len(val_target):
            raise ValueError(
                "Validation target exists but no held-out selection mask was created"
            )
    elif val_selection_mask.sum() == 0:
        raise ValueError(
            "Validation split has no observed target positions for selection-HO validation"
        )

    train_dataset = WindowedTimeSeriesDataset(
        train_target_scaled,
        train_aux_scaled,
        config.window_size,
        config.stride,
        mode="train",
        denoise_prob=config.denoise_prob,
        seed=config.seed,
        aux_mask=train_aux_mask,
        aux_mask_channel=preprocessing.aux_mask_channel,
        dynamic_mask_config=dynamic_mask_config,
        fixed_mask=train_fixed_mask,
        censor_mask=train_censor,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training split ({len(train_target)} rows) is shorter than window_size={config.window_size}"
        )

    train_ho_dataset = None
    if config.train_ho_enabled:
        train_ho_dataset = WindowedTimeSeriesDataset(
            train_target_scaled,
            train_aux_scaled,
            config.window_size,
            config.stride,
            mode="val",
            seed=config.seed,
            aux_mask=train_aux_mask,
            aux_mask_channel=preprocessing.aux_mask_channel,
            selection_mask=train_selection_mask,
            censor_mask=train_censor,
        )
        if len(train_ho_dataset) == 0:
            raise ValueError("Training split produced no windows for train-HO evaluation")

    val_dataset = WindowedTimeSeriesDataset(
        val_target_scaled,
        val_aux_scaled,
        config.window_size,
        config.stride,
        mode="val",
        seed=config.seed,
        aux_mask=val_aux_mask,
        aux_mask_channel=preprocessing.aux_mask_channel,
        selection_mask=val_selection_mask,
        censor_mask=val_censor,
    )
    if len(val_target) and config.val_fraction > 0 and len(val_dataset) == 0:
        raise ValueError(
            f"Validation split ({len(val_target)} rows) is shorter than window_size={config.window_size}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        **_loader_options(config, config.train_loader_num_workers),
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_ho_loader = (
        DataLoader(
            train_ho_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **_loader_options(config, config.val_loader_num_workers),
        )
        if train_ho_dataset is not None
        else None
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **_loader_options(config, config.val_loader_num_workers),
        )
        if len(val_dataset) > 0
        else None
    )

    model = ImputationVAE_Graph(
        target_dim=data_schema.target_dim,
        aux_dim=(2 if preprocessing.aux_mask_channel else 1) * data_schema.aux_dim,
        window_size=config.window_size,
        **model_kwargs,
    )

    device = setup_device()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[graph-temporal-vae] {len(frame)} rows -> {len(train_dataset)} train / "
        f"{len(val_dataset) if val_loader is not None else 0} val windows "
        f"({len(train_loader)} batches/epoch), {data_schema.target_dim} targets "
        f"({n_chem} chem + {len(data_schema.psd_cols)} psd), {data_schema.aux_dim} met cols, "
        f"{n_params:,} params, device={device}, epochs={config.epochs}, batch_size={config.batch_size}"
    )
    if train_selection_mask is not None:
        print(
            f"[graph-temporal-vae] train-HO: {int(train_selection_mask.sum())} "
            f"observed cells held out with seed={config.train_ho_seed}, "
            f"ratio={config.train_ho_ratio}"
        )
    if selection_full is not None:
        print(
            f"[graph-temporal-vae] global-HO: {int(selection_full.sum())} "
            f"observed cells excluded from stage-one input/loss with "
            f"seed={config.selection_val_seed}, ratio={config.selection_mask_ratio}"
        )
    if censoring.active:
        fractions = censor_report["fractions"]
        print(
            f"[graph-temporal-vae] censoring: {fractions['censored'] * 100:.1f}% of target "
            f"cells below detection limit across {censor_report['n_censored_columns']} column(s), "
            f"{fractions['observed'] * 100:.1f}% observed, {fractions['missing'] * 100:.1f}% missing "
            f"(loss={censoring.loss}, input_fill={censoring.input_fill})"
        )
        saturated = high_censoring_columns(censor_report, 0.9)
        if saturated:
            print(
                f"[graph-temporal-vae] {len(saturated)} column(s) are >=90% non-detect and carry "
                f"little concentration signal; consider dropping them: {saturated}"
            )

    history_path = os.path.splitext(save_path)[0] + "_history.csv"
    trainer = Trainer(
        model, train_loader, val_loader, config, device,
        censor_threshold=censor_threshold_scaled,
        history_path=history_path,
        train_ho_loader=train_ho_loader,
    )
    best_val = trainer.fit()
    refit_summary = None
    refit_history_path = None
    refit_monitor_mask = None
    if config.full_data_refit_epochs > 0:
        # Snapshot the stage-1 (pre-refit) weights before refit touches them
        # any further -- this is the only way to later ask "how much did
        # stage 2 change predictions", since refit trains the model in place
        # and nothing else preserves this exact state.
        stage1_bundle = _build_bundle(
            trainer=trainer, config=config, model_kwargs=model_kwargs,
            target_cols=target_cols, aux_cols=aux_cols, preprocessing=preprocessing,
            censoring=censoring, censor_threshold_scaled=censor_threshold_scaled,
            censor_report=censor_report, history_path=history_path,
            refit_history_path=None, refit_summary=None,
            selection_full=selection_full, best_val=best_val,
            selection_mask_diagnostics=selection_mask_diagnostics,
            data_schema=data_schema, data_interface=data_interface,
            scaler_target=scaler_target, scaler_aux=scaler_aux,
        )
        stage1_save_path = os.path.splitext(save_path)[0] + "_stage1.pt"
        stage1_out_dir = os.path.dirname(stage1_save_path)
        if stage1_out_dir:
            os.makedirs(stage1_out_dir, exist_ok=True)
        torch.save(stage1_bundle, stage1_save_path)
        # Stage two restores the global-HO cells to the loss and trains on the
        # full timeline. A separate deterministic mask with the same shape
        # family as dynamic training masks provides a stable, task-aligned
        # monitor. It is not independent validation because its target values
        # are included in full-data training.
        full_target_scaled = scaler_target.transform(target_model_space)
        full_target_scaled[np.isnan(target_model_space)] = np.nan
        full_aux_scaled = (
            scaler_aux.transform(aux_model_space) if aux_cols else aux_model_space
        )
        full_aux_mask = ~np.isnan(aux_model_space)
        refit_monitor_seed = config.selection_val_seed + 2
        refit_monitor_mask = sample_block_heldout_mask_to_ratio(
            observed_full,
            {**dynamic_mask_config, "ensure_nonempty": True},
            seed=refit_monitor_seed,
        )
        if refit_monitor_mask.sum() == 0:
            raise ValueError(
                "Full-data refit monitor produced no observed held-out cells"
            )

        refit_dataset = WindowedTimeSeriesDataset(
            full_target_scaled,
            full_aux_scaled,
            config.window_size,
            config.stride,
            mode="train",
            denoise_prob=config.denoise_prob,
            seed=config.seed + 1,
            aux_mask=full_aux_mask,
            aux_mask_channel=preprocessing.aux_mask_channel,
            dynamic_mask_config=dynamic_mask_config,
            fixed_mask=None,
            censor_mask=censor_full,
        )
        refit_monitor_dataset = WindowedTimeSeriesDataset(
            full_target_scaled,
            full_aux_scaled,
            config.window_size,
            config.stride,
            mode="val",
            seed=config.seed + 1,
            aux_mask=full_aux_mask,
            aux_mask_channel=preprocessing.aux_mask_channel,
            selection_mask=refit_monitor_mask,
            censor_mask=censor_full,
        )
        refit_loader = DataLoader(
            refit_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            **_loader_options(config, config.train_loader_num_workers),
            generator=torch.Generator().manual_seed(config.seed + 1),
        )
        refit_monitor_loader = DataLoader(
            refit_monitor_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            **_loader_options(config, config.val_loader_num_workers),
        )
        refit_history_path = os.path.splitext(save_path)[0] + "_refit_history.csv"
        print(
            "[graph-temporal-vae] full-data refit: restored "
            f"{int(selection_full.sum())} global-HO cells, "
            f"max_epochs={config.full_data_refit_epochs}, "
            f"dynamic-HO monitor cells={int(refit_monitor_mask.sum())}, "
            f"monitor_seed={refit_monitor_seed}"
        )
        refit_summary = trainer.refit_full_data(
            refit_loader,
            refit_monitor_loader,
            history_path=refit_history_path,
        )
        refit_summary["monitor_seed"] = refit_monitor_seed
        refit_summary["monitor_cells"] = int(refit_monitor_mask.sum())

    bundle = _build_bundle(
        trainer=trainer, config=config, model_kwargs=model_kwargs,
        target_cols=target_cols, aux_cols=aux_cols, preprocessing=preprocessing,
        censoring=censoring, censor_threshold_scaled=censor_threshold_scaled,
        censor_report=censor_report, history_path=history_path,
        refit_history_path=refit_history_path, refit_summary=refit_summary,
        selection_full=selection_full, best_val=best_val,
        selection_mask_diagnostics=selection_mask_diagnostics,
        data_schema=data_schema, data_interface=data_interface,
        scaler_target=scaler_target, scaler_aux=scaler_aux,
    )
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(bundle, save_path)
    return best_val
