"""
Masking Utilities for Synthetic Missing Data Generation
========================================================
Reusable module for generating synthetic missing patterns with
normally-distributed gap lengths. Extracted from missing_pattern_experiment_beijing.py.
"""
import numpy as np
import json
import os


def variable_length_gaps(n_samples, n_features, total_ratio,
                         mean_duration=48, std_duration=24,
                         min_duration=6, max_duration=168, seed=42):
    """
    Generate synthetic missing mask with variable-length continuous gaps.
    
    Gap durations follow N(mean_duration, std_duration²), clipped to [min_duration, max_duration].
    Gaps are placed per-feature independently to achieve the target missing ratio.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        total_ratio: Target fraction of data to mask (e.g. 0.30 for 30%)
        mean_duration: Mean gap length in hours (default 48)
        std_duration: Std dev of gap length in hours (default 24)
        min_duration: Minimum gap length (default 6)
        max_duration: Maximum gap length (default 168)
        seed: Random seed for reproducibility
        
    Returns:
        mask: np.ndarray [n_samples, n_features], 1=observed, 0=masked
        gap_info: list of lists, per-feature gap tuples (start, end, duration)
    """
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.ones((n_samples, n_features), dtype=float)
    gap_info = [[] for _ in range(n_features)]
    
    for feature_idx in range(n_features):
        current_missing = 0
        target_missing = int(n_samples * total_ratio)
        position = 0
        
        while current_missing < target_missing and position < n_samples:
            # Sample gap duration from normal distribution
            duration = np.random.normal(mean_duration, std_duration)
            duration = int(np.clip(duration, min_duration, max_duration))
            
            # Sample interval between gaps
            mean_interval = int(mean_duration * (1 - total_ratio) / total_ratio)
            std_interval = mean_interval * 0.5
            interval = int(np.random.normal(mean_interval, std_interval))
            interval = max(min_duration, interval)
            
            position += interval
            
            if position + duration > n_samples:
                duration = n_samples - position
                if duration < min_duration:
                    break
            
            end = min(position + duration, n_samples)
            actual_duration = end - position
            mask[position:end, feature_idx] = 0.0
            gap_info[feature_idx].append((int(position), int(end), int(actual_duration)))
            
            current_missing += actual_duration
            position = end
            
            if current_missing >= target_missing:
                break
    
    return mask, gap_info


def save_mask(mask, gap_info, output_dir, config=None):
    """
    Save synthetic mask and metadata to disk.
    
    Saves:
        - synthetic_mask.npy: Binary mask array [T, F]
        - mask_info.json: Gap statistics and configuration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mask array
    np.save(os.path.join(output_dir, 'synthetic_mask.npy'), mask)
    
    # Compute gap statistics
    all_durations = [dur for feature_gaps in gap_info for (_, _, dur) in feature_gaps]
    
    info = {
        'actual_missing_ratio': float(1 - mask.mean()),
        'mask_shape': list(mask.shape),
        'n_gaps_total': len(all_durations),
    }
    
    if all_durations:
        info['gap_duration_mean'] = float(np.mean(all_durations))
        info['gap_duration_std'] = float(np.std(all_durations))
        info['gap_duration_min'] = int(np.min(all_durations))
        info['gap_duration_max'] = int(np.max(all_durations))
    
    if config:
        info['config'] = config
    
    with open(os.path.join(output_dir, 'mask_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"  ✅ Saved synthetic_mask.npy ({mask.shape}) and mask_info.json")
    print(f"     Actual missing ratio: {info['actual_missing_ratio']:.2%}")
    if all_durations:
        print(f"     Gap stats: μ={info['gap_duration_mean']:.1f}hr, "
              f"σ={info['gap_duration_std']:.1f}hr, "
              f"range=[{info['gap_duration_min']}-{info['gap_duration_max']}]hr, "
              f"n_gaps={info['n_gaps_total']}")
    
    return info


def load_mask(output_dir):
    """
    Load a previously saved synthetic mask.
    
    Returns:
        mask: np.ndarray [T, F]
        info: dict with mask metadata
    """
    mask = np.load(os.path.join(output_dir, 'synthetic_mask.npy'))
    with open(os.path.join(output_dir, 'mask_info.json'), 'r') as f:
        info = json.load(f)
    return mask, info


def apply_synthetic_mask(df, target_cols, mask):
    """
    Apply a synthetic mask to a DataFrame, setting masked positions to NaN.
    
    Args:
        df: DataFrame with target columns
        target_cols: list of column names to mask
        mask: np.ndarray [T, F], 1=observed, 0=masked
        
    Returns:
        df_masked: DataFrame with synthetic NaNs applied
    """
    df_masked = df.copy()
    for i, col in enumerate(target_cols):
        if col in df_masked.columns:
            df_masked.loc[mask[:, i] == 0, col] = np.nan
    return df_masked
