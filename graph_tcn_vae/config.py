"""Configuration for the reference train/impute pipeline."""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainConfig:
    csv: List[str]
    timestamp_col: str
    target_cols: List[str]
    aux_cols: List[str] = field(default_factory=list)
    target_transform: str = "none"

    window_size: int = 48
    stride: int = 24
    val_fraction: float = 0.15

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 15
    # Legacy point-drop augmentation. Dynamic contiguous HO masking is the
    # default training protocol; this can be kept at zero unless an additional
    # random point-drop signal is explicitly desired.
    denoise_prob: float = 0.0
    dynamic_mask_target_ratio: float = 0.10
    dynamic_mask_mean_duration: float = 48.0
    dynamic_mask_std_duration: float = 24.0
    dynamic_mask_min_duration: int = 3
    dynamic_mask_max_duration: int = 168
    dynamic_mask_chem_blocks: int = 1
    dynamic_mask_psd_blocks: int = 1
    selection_val_seed: int = 100003
    validation_metric: str = "ho_nll"
    val_crps_mc_samples: int = 20
    val_crps_every_n_epochs: int = 1
    kl_warmup_epochs: Optional[int] = None
    kl_max_beta: float = 1.0
    seed: int = 0

    # Passthrough to ImputationVAE_Graph(...) beyond target_dim/aux_dim/window_size,
    # which are derived from the data and set automatically.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.csv, str):
            self.csv = [self.csv]
        for derived in ("target_dim", "aux_dim", "window_size"):
            self.model_kwargs.pop(derived, None)
        if self.validation_metric not in {"ho_nll", "ho_mse", "ho_crps"}:
            raise ValueError("validation_metric must be 'ho_nll', 'ho_mse', or 'ho_crps'")
        if self.target_transform not in {"none", "log1p"}:
            raise ValueError("target_transform must be 'none' or 'log1p'")
        if self.window_size < 1 or self.stride < 1:
            raise ValueError("window_size and stride must be positive")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.val_crps_mc_samples < 2 or self.val_crps_every_n_epochs < 1:
            raise ValueError("CRPS validation requires at least 2 MC samples and a positive interval")

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self):
        return asdict(self)
