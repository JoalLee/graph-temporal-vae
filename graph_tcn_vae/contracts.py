"""Versioned public contracts for multimodal aerosol data and preprocessing.

The high-level package interface treats each measurement modality as a file
role rather than requiring callers to enumerate every column manually.
Column names and order discovered during training are persisted in
:class:`DataSchema` and enforced during inference.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SUPPORTED_TRANSFORMS = {"none", "log1p"}
SUPPORTED_SCALERS = {"standard", "robust", "minmax", "none"}


def _normalize_paths(values: Optional[Sequence[str]]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        return [str(values)]
    return [str(value) for value in values if str(value)]


@dataclass
class ModalityFiles:
    """Runtime CSV paths grouped by measurement modality.

    At least one target modality (``chemistry`` or ``psd``) is required.
    ``meteorology`` is optional conditioning data.
    """

    chemistry: List[str] = field(default_factory=list)
    psd: List[str] = field(default_factory=list)
    meteorology: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.chemistry = _normalize_paths(self.chemistry)
        self.psd = _normalize_paths(self.psd)
        self.meteorology = _normalize_paths(self.meteorology)
        if not self.chemistry and not self.psd:
            raise ValueError("At least one target modality CSV is required: chemistry or psd")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModalityFiles":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def all_paths(self) -> List[str]:
        return [*self.chemistry, *self.psd, *self.meteorology]


@dataclass
class ModalityPreprocessing:
    """Transform and affine scaling applied to one modality."""

    transform: str = "none"
    scaler: str = "standard"
    output_transform: Optional[str] = None

    def __post_init__(self):
        if self.transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {sorted(SUPPORTED_TRANSFORMS)}, got {self.transform!r}"
            )
        if self.scaler not in SUPPORTED_SCALERS:
            raise ValueError(
                f"scaler must be one of {sorted(SUPPORTED_SCALERS)}, got {self.scaler!r}"
            )
        if self.output_transform is None:
            self.output_transform = self.transform
        if self.output_transform not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                "output_transform must be one of "
                f"{sorted(SUPPORTED_TRANSFORMS)}, got {self.output_transform!r}"
            )

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "ModalityPreprocessing":
        return cls(**values)


@dataclass
class PreprocessingConfig:
    """Preprocessing choices for each public modality.

    ``fit_scope='train'`` is the leakage-safe default. ``'full'`` is retained
    for exact reproduction of the private 26e reference preprocessing.
    """

    chemistry: ModalityPreprocessing = field(default_factory=ModalityPreprocessing)
    psd: ModalityPreprocessing = field(default_factory=ModalityPreprocessing)
    meteorology: ModalityPreprocessing = field(default_factory=ModalityPreprocessing)
    fit_scope: str = "train"
    aux_mask_channel: bool = True

    def __post_init__(self):
        for name in ("chemistry", "psd", "meteorology"):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, ModalityPreprocessing.from_dict(value))
        if self.fit_scope not in {"train", "full"}:
            raise ValueError("fit_scope must be 'train' or 'full'")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "PreprocessingConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataSchema:
    """Resolved, versioned column contract stored in every new bundle."""

    schema_version: int = 1
    timestamp_col: str = "time"
    chemistry_cols: List[str] = field(default_factory=list)
    psd_cols: List[str] = field(default_factory=list)
    meteorology_cols: List[str] = field(default_factory=list)
    frequency: Optional[str] = None
    timezone: Optional[str] = None
    time_grid_policy: str = "strict"
    duplicate_timestamp_policy: str = "error"
    psd_diameters_nm: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.chemistry_cols = list(self.chemistry_cols)
        self.psd_cols = list(self.psd_cols)
        self.meteorology_cols = list(self.meteorology_cols)
        self.psd_diameters_nm = [float(value) for value in self.psd_diameters_nm]
        if not self.chemistry_cols and not self.psd_cols:
            raise ValueError("DataSchema requires at least one chemistry or PSD column")
        all_columns = [*self.chemistry_cols, *self.psd_cols, *self.meteorology_cols]
        duplicates = sorted({column for column in all_columns if all_columns.count(column) > 1})
        if duplicates:
            raise ValueError(f"Columns cannot appear in multiple modalities: {duplicates}")
        if self.psd_diameters_nm:
            if len(self.psd_diameters_nm) != len(self.psd_cols):
                raise ValueError("psd_diameters_nm must match psd_cols length")
            if any(b <= a for a, b in zip(self.psd_diameters_nm, self.psd_diameters_nm[1:])):
                raise ValueError("PSD columns must be ordered by strictly increasing diameter")
        elif not self.psd_cols:
            self.psd_diameters_nm = []
        if not self.psd_cols and self.psd_diameters_nm:
            raise ValueError("psd_diameters_nm must be empty when no PSD columns are present")
        if self.time_grid_policy not in {"strict", "reindex", "row_order"}:
            raise ValueError("time_grid_policy must be 'strict', 'reindex', or 'row_order'")
        if self.duplicate_timestamp_policy not in {"error", "first"}:
            raise ValueError("duplicate_timestamp_policy must be 'error' or 'first'")

    @property
    def target_cols(self) -> List[str]:
        return [*self.chemistry_cols, *self.psd_cols]

    @property
    def auxiliary_cols(self) -> List[str]:
        return list(self.meteorology_cols)

    @property
    def n_chem(self) -> int:
        return len(self.chemistry_cols)

    @property
    def target_dim(self) -> int:
        return len(self.target_cols)

    @property
    def aux_dim(self) -> int:
        return len(self.meteorology_cols)

    @property
    def modality_slices(self) -> Dict[str, Tuple[int, int]]:
        n_chem = len(self.chemistry_cols)
        return {
            "chemistry": (0, n_chem),
            "psd": (n_chem, n_chem + len(self.psd_cols)),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "DataSchema":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceConfig:
    """Validated runtime controls for overlap inference."""

    stride: Optional[int] = None
    n_mc_samples: int = 50
    inference_batch_size: int = 4
    mc_batch_size: int = 1
    interval_lower: float = 0.05
    interval_upper: float = 0.95
    support_context_window: int = 72

    def __post_init__(self):
        if self.stride is not None and self.stride < 1:
            raise ValueError("stride must be positive when provided")
        if self.n_mc_samples < 2:
            raise ValueError("n_mc_samples must be at least 2")
        if self.inference_batch_size < 1 or self.mc_batch_size < 1:
            raise ValueError("inference and MC batch sizes must be positive")
        if not 0.0 < self.interval_lower < self.interval_upper < 1.0:
            raise ValueError("interval bounds must satisfy 0 < lower < upper < 1")
        if self.support_context_window < 1:
            raise ValueError("support_context_window must be positive")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "InferenceConfig":
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
