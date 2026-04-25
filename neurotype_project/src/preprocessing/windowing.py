"""
AlterEgo – Feature Extraction (windowing.py)

Extracts the canonical 40-feature vector from a (WINDOW_SIZE × N_CHANNELS)
raw EMG window.

Feature set (5 per channel × 8 channels = 40 total):
  1. RMS   – Root Mean Square
  2. MAV   – Mean Absolute Value
  3. VAR   – Signal Variance
  4. ZCR   – Zero Crossing Rate
  5. SSC   – Slope Sign Change
"""

import numpy as np
from src.config import (
    N_CHANNELS, WINDOW_SIZE, N_FEATURES,
    ZCR_THRESHOLD, SSC_THRESHOLD
)


# ──────────────────────────────────────────────────────────────
# Individual feature functions (vectorised over channel axis)
# ──────────────────────────────────────────────────────────────

def compute_rms(window: np.ndarray) -> np.ndarray:
    """Root Mean Square – energy estimator.
    Args:
        window: (WINDOW_SIZE, N_CHANNELS)
    Returns:
        (N_CHANNELS,)
    """
    return np.sqrt(np.mean(window ** 2, axis=0))


def compute_mav(window: np.ndarray) -> np.ndarray:
    """Mean Absolute Value – amplitude estimator.
    Args:
        window: (WINDOW_SIZE, N_CHANNELS)
    Returns:
        (N_CHANNELS,)
    """
    return np.mean(np.abs(window), axis=0)


def compute_variance(window: np.ndarray) -> np.ndarray:
    """Signal Variance – spread estimator.
    Args:
        window: (WINDOW_SIZE, N_CHANNELS)
    Returns:
        (N_CHANNELS,)
    """
    return np.var(window, axis=0)


def compute_zcr(window: np.ndarray, threshold: float = ZCR_THRESHOLD) -> np.ndarray:
    """Zero Crossing Rate – frequency-domain proxy.

    Formula (matches Kaggle training pipeline):
        np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0) / len(window)

    Args:
        window:    (WINDOW_SIZE, N_CHANNELS)
        threshold: (unused – kept for API compatibility)
    Returns:
        (N_CHANNELS,)
    """
    return np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0) / len(window)


def compute_ssc(window: np.ndarray, threshold: float = SSC_THRESHOLD) -> np.ndarray:
    """Slope Sign Change – another frequency proxy.

    Formula (matches Kaggle training pipeline):
        np.sum(np.diff(np.sign(np.diff(window, axis=0)), axis=0) != 0, axis=0)

    Args:
        window:    (WINDOW_SIZE, N_CHANNELS)
        threshold: (unused – kept for API compatibility)
    Returns:
        (N_CHANNELS,)
    """
    return np.sum(
        np.diff(np.sign(np.diff(window, axis=0)), axis=0) != 0,
        axis=0,
    ).astype(float)


# ──────────────────────────────────────────────────────────────
# Combined extractor
# ──────────────────────────────────────────────────────────────

def extract_features(window: np.ndarray) -> np.ndarray:
    """Extract the 40-feature vector from a raw EMG window.

    Args:
        window: numpy array of shape (WINDOW_SIZE, N_CHANNELS)
                Values should be in physical units (mV) or at least
                consistently scaled.

    Returns:
        feature_vector: numpy array of shape (N_FEATURES,)  i.e. (40,)
                        organised as:
                        [ch0_rms, ch1_rms, …, ch7_rms,
                         ch0_mav, …, ch7_mav,
                         ch0_var, …, ch7_var,
                         ch0_zcr, …, ch7_zcr,
                         ch0_ssc, …, ch7_ssc]

    Raises:
        ValueError: if the window shape does not match expectations.
    """
    if window.ndim != 2:
        raise ValueError(f"Expected 2-D window, got shape {window.shape}")
    if window.shape[1] != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} channels, got {window.shape[1]}"
        )

    rms = compute_rms(window)           # (8,)
    mav = compute_mav(window)           # (8,)
    var = compute_variance(window)      # (8,)
    zcr = compute_zcr(window)           # (8,)
    ssc = compute_ssc(window)           # (8,)

    feature_vector = np.concatenate([rms, mav, var, zcr, ssc])  # (40,)
    assert feature_vector.shape == (N_FEATURES,), (
        f"Feature vector shape mismatch: {feature_vector.shape}"
    )
    return feature_vector


# ──────────────────────────────────────────────────────────────
# Sliding-window generator (used during calibration data capture)
# ──────────────────────────────────────────────────────────────

def sliding_window_features(
    signal:     np.ndarray,
    window_size: int   = WINDOW_SIZE,
    overlap:     float = 0.5,
) -> np.ndarray:
    """Slide a feature extraction window over a longer signal.

    Args:
        signal:      (total_samples, N_CHANNELS)
        window_size: samples per window
        overlap:     fraction of overlap between consecutive windows [0, 1)

    Returns:
        features: (n_windows, N_FEATURES)
    """
    step = max(1, int(window_size * (1 - overlap)))
    starts = range(0, signal.shape[0] - window_size + 1, step)
    features = [extract_features(signal[s: s + window_size]) for s in starts]
    return np.array(features, dtype=np.float32)