"""Feature transforms and NaN-aware affine scaling for multimodal inputs."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .contracts import DataSchema, ModalityPreprocessing, PreprocessingConfig


def transform_values(array, transform="none", *, label="values"):
    values = np.asarray(array, dtype=np.float64)
    if transform == "none":
        return values.copy()
    if transform != "log1p":
        raise ValueError(f"Unsupported transform {transform!r}")
    finite = np.isfinite(values)
    if np.any(values[finite] < 0):
        raise ValueError(f"{label} transform='log1p' requires non-negative finite values")
    out = values.copy()
    out[finite] = np.log1p(out[finite])
    return out


def inverse_values(array, transform="none"):
    values = np.asarray(array, dtype=np.float64)
    if transform == "none":
        return values.copy()
    if transform != "log1p":
        raise ValueError(f"Unsupported transform {transform!r}")
    return np.expm1(values).clip(min=0.0)


class NaNAwareAffineScaler:
    """Per-feature affine scaler supporting standard, robust, minmax, or none.

    All scaler modes share the same transform ``(x - center) / scale``.  The
    fitted center and scale are persisted, so inference never re-fits them.
    ``mean_``/``std_`` aliases are retained for compatibility with older code.
    """

    def __init__(self, kind="standard"):
        if kind not in {"standard", "robust", "minmax", "none", "mixed"}:
            raise ValueError(f"Unsupported scaler kind {kind!r}")
        self.kind = kind
        self.center_ = None
        self.scale_ = None
        self.feature_kinds_ = None

    @property
    def mean_(self):
        return self.center_

    @mean_.setter
    def mean_(self, value):
        self.center_ = value

    @property
    def std_(self):
        return self.scale_

    @std_.setter
    def std_(self, value):
        self.scale_ = value

    def fit(self, array):
        values = np.asarray(array, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Scaler input must be 2-D, got shape {values.shape}")
        if values.shape[1] == 0:
            self.center_ = np.zeros(0, dtype=np.float64)
            self.scale_ = np.ones(0, dtype=np.float64)
            self.feature_kinds_ = []
            return self
        all_nan = np.isnan(values).all(axis=0)
        if all_nan.any():
            indices = np.flatnonzero(all_nan).tolist()
            raise ValueError(f"Cannot fit scaler: all values are missing in column indices {indices}")

        if self.kind == "standard":
            center = np.nanmean(values, axis=0)
            scale = np.nanstd(values, axis=0)
        elif self.kind == "robust":
            center = np.nanmedian(values, axis=0)
            q25 = np.nanpercentile(values, 25.0, axis=0)
            q75 = np.nanpercentile(values, 75.0, axis=0)
            scale = q75 - q25
        elif self.kind == "minmax":
            center = np.nanmin(values, axis=0)
            scale = np.nanmax(values, axis=0) - center
        elif self.kind == "none":
            center = np.zeros(values.shape[1], dtype=np.float64)
            scale = np.ones(values.shape[1], dtype=np.float64)
        else:
            raise ValueError("kind='mixed' can only be created with concatenate")

        self.center_ = np.asarray(center, dtype=np.float64)
        self.scale_ = np.where(np.asarray(scale, dtype=np.float64) < 1e-8, 1.0, scale)
        self.feature_kinds_ = [self.kind] * values.shape[1]
        return self

    def transform(self, array):
        values = np.asarray(array, dtype=np.float64)
        scaled = (values - self.center_) / self.scale_
        return np.nan_to_num(scaled, nan=0.0)

    def inverse_transform(self, array):
        return np.asarray(array, dtype=np.float64) * self.scale_ + self.center_

    def inverse_transform_std(self, std_array):
        return np.asarray(std_array, dtype=np.float64) * self.scale_

    def to_dict(self):
        return {
            "kind": self.kind,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "feature_kinds": list(self.feature_kinds_ or []),
            # Legacy keys allow old loaders and external scripts to inspect the
            # affine statistics without understanding the newer scaler schema.
            "mean": self.center_.tolist(),
            "std": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, state):
        kind = state.get("kind", "standard")
        scaler = cls(kind=kind)
        center = state.get("center")
        if center is None:
            center = state["mean"]
        scale = state.get("scale")
        if scale is None:
            scale = state["std"]
        scaler.center_ = np.asarray(center, dtype=np.float64)
        scaler.scale_ = np.asarray(scale, dtype=np.float64)
        scaler.feature_kinds_ = list(state.get("feature_kinds", [kind] * len(scaler.center_)))
        return scaler

    @classmethod
    def concatenate(cls, scalers: Sequence["NaNAwareAffineScaler"]):
        scalers = list(scalers)
        result = cls(kind="mixed")
        if not scalers:
            result.center_ = np.zeros(0, dtype=np.float64)
            result.scale_ = np.ones(0, dtype=np.float64)
            result.feature_kinds_ = []
            return result
        result.center_ = np.concatenate([scaler.center_ for scaler in scalers])
        result.scale_ = np.concatenate([scaler.scale_ for scaler in scalers])
        result.feature_kinds_ = [
            kind for scaler in scalers for kind in (scaler.feature_kinds_ or [scaler.kind] * len(scaler.center_))
        ]
        return result


@dataclass
class PreparedArrays:
    target_model_space: np.ndarray
    auxiliary_model_space: np.ndarray
    target_scaler: NaNAwareAffineScaler
    auxiliary_scaler: NaNAwareAffineScaler
    target_output_transforms: List[str]


def _target_specs(schema: DataSchema, config: PreprocessingConfig):
    specs: List[Tuple[slice, ModalityPreprocessing, str]] = []
    chem_end = len(schema.chemistry_cols)
    if chem_end:
        specs.append((slice(0, chem_end), config.chemistry, "chemistry"))
    if schema.psd_cols:
        specs.append((slice(chem_end, chem_end + len(schema.psd_cols)), config.psd, "psd"))
    return specs


def transform_targets(array, schema: DataSchema, config: PreprocessingConfig):
    values = np.asarray(array, dtype=np.float64)
    if values.shape[-1] != schema.target_dim:
        raise ValueError(
            f"Target feature dimension {values.shape[-1]} does not match schema {schema.target_dim}"
        )
    out = values.copy()
    for feature_slice, spec, label in _target_specs(schema, config):
        out[..., feature_slice] = transform_values(
            out[..., feature_slice], spec.transform, label=label
        )
    return out


def inverse_targets(array, schema: DataSchema, config: PreprocessingConfig):
    values = np.asarray(array, dtype=np.float64)
    if values.shape[-1] != schema.target_dim:
        raise ValueError(
            f"Target feature dimension {values.shape[-1]} does not match schema {schema.target_dim}"
        )
    out = values.copy()
    for feature_slice, spec, _label in _target_specs(schema, config):
        out[..., feature_slice] = inverse_values(
            out[..., feature_slice], spec.output_transform
        )
    return out


def target_output_transforms(schema: DataSchema, config: PreprocessingConfig) -> List[str]:
    return (
        [config.chemistry.output_transform] * len(schema.chemistry_cols)
        + [config.psd.output_transform] * len(schema.psd_cols)
    )


def observed_targets_to_output(array, schema: DataSchema, config: PreprocessingConfig):
    """Map CSV values through the same input/output transform contract.

    Physical input with ``transform=log1p, output_transform=log1p`` therefore
    remains unchanged, while an already-log1p CSV configured as
    ``transform=none, output_transform=log1p`` is correctly exponentiated.
    """
    model_space = transform_targets(array, schema, config)
    return inverse_targets(model_space, schema, config)


def transform_auxiliary(array, config: PreprocessingConfig):
    return transform_values(array, config.meteorology.transform, label="meteorology")


def fit_target_scaler(
    fit_values,
    schema: DataSchema,
    config: PreprocessingConfig,
) -> NaNAwareAffineScaler:
    fit_values = np.asarray(fit_values, dtype=np.float64)
    scalers = []
    for feature_slice, spec, _label in _target_specs(schema, config):
        scalers.append(NaNAwareAffineScaler(spec.scaler).fit(fit_values[:, feature_slice]))
    return NaNAwareAffineScaler.concatenate(scalers)


def fit_auxiliary_scaler(fit_values, config: PreprocessingConfig) -> NaNAwareAffineScaler:
    fit_values = np.asarray(fit_values, dtype=np.float64)
    if fit_values.shape[1] == 0:
        return NaNAwareAffineScaler("none").fit(np.zeros((1, 0)))
    return NaNAwareAffineScaler(config.meteorology.scaler).fit(fit_values)


def preprocessing_from_legacy(
    target_transform="none",
    target_output_transform=None,
    scaler_fit_scope="train",
    aux_mask_channel=True,
) -> PreprocessingConfig:
    target_output_transform = target_output_transform or target_transform
    target_spec = ModalityPreprocessing(
        transform=target_transform,
        scaler="standard",
        output_transform=target_output_transform,
    )
    return PreprocessingConfig(
        chemistry=target_spec,
        psd=ModalityPreprocessing(
            transform=target_transform,
            scaler="standard",
            output_transform=target_output_transform,
        ),
        meteorology=ModalityPreprocessing(transform="none", scaler="standard"),
        fit_scope=scaler_fit_scope,
        aux_mask_channel=aux_mask_channel,
    )
