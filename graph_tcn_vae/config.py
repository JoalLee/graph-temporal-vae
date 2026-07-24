"""Configuration for the reference train/impute pipeline."""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import ModalityFiles, PreprocessingConfig
from .model_config import ModelConfig
from .preprocessing import preprocessing_from_legacy


@dataclass
class TrainConfig:
    # Legacy single/merged-CSV interface. New callers should prefer
    # ``modality_files`` so Chem, PSD, and meteorology are explicit roles.
    csv: List[str] = field(default_factory=list)
    timestamp_col: str = "time"
    target_cols: List[str] = field(default_factory=list)
    aux_cols: List[str] = field(default_factory=list)
    modality_files: Optional[ModalityFiles] = None
    preprocessing: Optional[PreprocessingConfig] = None
    # Transform applied to values read from the CSV before scaling/training.
    # For an already-log1p-preprocessed experiment CSV, use "none" here.
    target_transform: str = "none"
    # Transform used to map model-space predictions back to output values.
    # None keeps backward-compatible behavior: it follows target_transform.
    target_output_transform: Optional[str] = None
    # The reference 26e preprocessing fits normalization on the complete
    # time axis before applying the fixed held-out mask.  General users may
    # prefer the leakage-safe train-only default.
    scaler_fit_scope: str = "train"
    # Time-grid contract. strict rejects missing/off-grid rows; reindex inserts
    # missing grid rows as NaN; row_order preserves legacy row-based behavior.
    expected_frequency: Optional[str] = None
    time_grid_policy: str = "strict"
    duplicate_timestamp_policy: str = "error"

    window_size: int = 48
    stride: int = 24
    val_fraction: float = 0.15

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    lr_min: float = 1e-6
    weight_decay: float = 0.0
    patience: int = 15
    train_loader_num_workers: int = 0
    val_loader_num_workers: int = 0
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
    dynamic_masking_mode: str = "block"
    dynamic_random_point_drop_prob: float = 0.0
    selection_val_seed: int = 100003
    selection_mask_mode: str = "block"
    selection_mask_ratio: float = 0.10
    shared_full_heldout_mask: bool = False
    validation_metric: str = "ho_nll"
    val_crps_mc_samples: int = 20
    val_crps_every_n_epochs: int = 1
    val_crps_dist_type: str = "gaussian"
    val_crps_use_inference_epoch: bool = True
    val_mc_batch_size: int = 1
    # Linear warmup + cosine anneal runs for the whole schedule by default.
    # The 26e reference instead switches to ReduceLROnPlateau (monitoring
    # held-out MSE) once warmup ends; set use_adaptive_lr=True to match it.
    use_adaptive_lr: bool = False
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 10
    lr_reduce_threshold: float = 1e-4
    lr_reduce_cooldown: int = 2
    # LR warmup previously had no config knob at all -- it was hardcoded to
    # 5% of `epochs` in Trainer.__init__. The 26e reference protocol
    # preserves an ABSOLUTE warmup length (100 epochs) even when a run uses
    # a reduced epoch budget (e.g. 700 instead of 2000), so a ratio of the
    # *current* run's epochs is the wrong knob when reproducing it -- set
    # lr_warmup_epochs explicitly in that case instead of lr_warmup_ratio.
    lr_warmup_epochs: Optional[int] = None
    lr_warmup_ratio: Optional[float] = None
    kl_warmup_epochs: Optional[int] = None
    kl_warmup_ratio: Optional[float] = None
    kl_strategy: str = "cosine"
    kl_cycles: int = 4
    kl_cycle_ratio: float = 0.5
    kl_max_beta: float = 1.0
    use_amp: bool = False
    amp_dtype: str = "auto"
    prior_type: str = "gaussian"
    use_gnll: bool = True
    use_student_t_nll: bool = False
    loss_normalization: str = "observed_mean"
    use_family_balanced_loss: bool = False
    family_loss_chem_weight: float = 0.5
    family_loss_scale: str = "target_dim"
    # The 26e reference uses 12x Chem and 1x PSD feature weighting.
    chem_feature_weight: float = 1.0
    psd_feature_weight: float = 1.0
    aux_mask_channel: bool = True
    seed: int = 0

    # Passthrough to ImputationVAE_Graph(...) beyond target_dim/aux_dim/window_size,
    # which are derived from the data and set automatically.
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.csv, str):
            self.csv = [self.csv]
        self.csv = list(self.csv)
        self.target_cols = list(self.target_cols)
        self.aux_cols = list(self.aux_cols)
        if isinstance(self.modality_files, dict):
            self.modality_files = ModalityFiles.from_dict(self.modality_files)
        if isinstance(self.preprocessing, dict):
            self.preprocessing = PreprocessingConfig.from_dict(self.preprocessing)
        if self.modality_files is not None:
            if self.csv or self.target_cols or self.aux_cols:
                raise ValueError(
                    "Use either modality_files or the legacy csv/target_cols/aux_cols interface, not both"
                )
        else:
            if not self.csv:
                raise ValueError("At least one CSV path is required")
            if not self.target_cols:
                raise ValueError("At least one target column is required")
            overlap = sorted(set(self.target_cols) & set(self.aux_cols))
            if overlap:
                raise ValueError(f"Columns cannot be both target and auxiliary: {overlap}")
        for derived in ("target_dim", "aux_dim", "window_size"):
            self.model_kwargs.pop(derived, None)
        unknown_model_fields = set(self.model_kwargs) - ModelConfig.field_names()
        if unknown_model_fields:
            raise TypeError(
                f"model_kwargs contain unexpected field(s): {sorted(unknown_model_fields)}"
            )
        if self.validation_metric not in {"ho_nll", "ho_mse", "ho_crps"}:
            raise ValueError("validation_metric must be 'ho_nll', 'ho_mse', or 'ho_crps'")
        if self.val_crps_dist_type not in {"gaussian", "student_t"}:
            raise ValueError("val_crps_dist_type must be 'gaussian' or 'student_t'")
        if self.target_transform not in {"none", "log1p"}:
            raise ValueError("target_transform must be 'none' or 'log1p'")
        if self.target_output_transform is None:
            self.target_output_transform = self.target_transform
        if self.target_output_transform not in {"none", "log1p"}:
            raise ValueError("target_output_transform must be 'none' or 'log1p'")
        if self.preprocessing is None:
            self.preprocessing = preprocessing_from_legacy(
                target_transform=self.target_transform,
                target_output_transform=self.target_output_transform,
                scaler_fit_scope=self.scaler_fit_scope,
                aux_mask_channel=self.aux_mask_channel,
            )
        else:
            self.scaler_fit_scope = self.preprocessing.fit_scope
            self.aux_mask_channel = self.preprocessing.aux_mask_channel
        if self.scaler_fit_scope not in {"train", "full"}:
            raise ValueError("scaler_fit_scope must be 'train' or 'full'")
        if self.time_grid_policy not in {"strict", "reindex", "row_order"}:
            raise ValueError(
                "time_grid_policy must be 'strict', 'reindex', or 'row_order'"
            )
        if self.duplicate_timestamp_policy not in {"error", "first"}:
            raise ValueError(
                "duplicate_timestamp_policy must be 'error' or 'first'"
            )
        if self.prior_type not in {"gaussian", "laplace", "student_t"}:
            raise ValueError("prior_type must be 'gaussian', 'laplace', or 'student_t'")
        if self.loss_normalization not in {"observed_mean", "window_feature_sum"}:
            raise ValueError(
                "loss_normalization must be 'observed_mean' or 'window_feature_sum'"
            )
        if self.amp_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("amp_dtype must be auto, bfloat16, float16, or float32")
        if self.dynamic_masking_mode not in {"block", "legacy"}:
            raise ValueError("dynamic_masking_mode must be 'block' or 'legacy'")
        if not 0 <= self.dynamic_random_point_drop_prob <= 1:
            raise ValueError("dynamic_random_point_drop_prob must be in [0, 1]")
        if self.selection_mask_mode not in {"block", "anchor_constrained"}:
            raise ValueError("selection_mask_mode must be 'block' or 'anchor_constrained'")
        if not 0 < self.selection_mask_ratio <= 1:
            raise ValueError("selection_mask_ratio must be in (0, 1]")
        if self.window_size < 1 or self.stride < 1:
            raise ValueError("window_size and stride must be positive")
        if self.stride > self.window_size:
            raise ValueError("stride cannot exceed window_size")
        if self.lr_min < 0 or self.weight_decay < 0:
            raise ValueError("lr_min and weight_decay must be non-negative")
        if self.chem_feature_weight < 0 or self.psd_feature_weight < 0:
            raise ValueError("feature weights must be non-negative")
        if self.chem_feature_weight == 0 and self.psd_feature_weight == 0:
            raise ValueError("at least one feature weight must be positive")
        if self.train_loader_num_workers < 0 or self.val_loader_num_workers < 0:
            raise ValueError("loader worker counts must be non-negative")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.val_crps_mc_samples < 2 or self.val_crps_every_n_epochs < 1:
            raise ValueError("CRPS validation requires at least 2 MC samples and a positive interval")
        if self.val_mc_batch_size < 1:
            raise ValueError("val_mc_batch_size must be positive")
        if self.kl_warmup_ratio is not None and not 0 < self.kl_warmup_ratio <= 1:
            raise ValueError("kl_warmup_ratio must be in (0, 1]")
        if self.lr_warmup_ratio is not None and not 0 < self.lr_warmup_ratio <= 1:
            raise ValueError("lr_warmup_ratio must be in (0, 1]")
        if self.lr_warmup_epochs is not None and self.lr_warmup_epochs < 1:
            raise ValueError("lr_warmup_epochs must be positive")
        if not 0 < self.lr_reduce_factor < 1:
            raise ValueError("lr_reduce_factor must be in (0, 1)")
        if self.lr_reduce_patience < 0 or self.lr_reduce_cooldown < 0:
            raise ValueError("lr_reduce_patience and lr_reduce_cooldown must be non-negative")
        if self.lr_reduce_threshold < 0:
            raise ValueError("lr_reduce_threshold must be non-negative")
        if self.kl_strategy not in {"linear", "cosine", "cyclical"}:
            raise ValueError("kl_strategy must be linear, cosine, or cyclical")
        if self.kl_cycles < 1 or not 0 < self.kl_cycle_ratio <= 1:
            raise ValueError("kl_cycles must be positive and kl_cycle_ratio in (0, 1]")

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self):
        return asdict(self)
