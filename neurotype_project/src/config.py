"""
AlterEgo - Central Configuration
Paths, Hyperparameters, and Constants
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models" / "kaggle"
MODEL_PATH   = MODELS_DIR / "emg_final_model_abc.h5"
SCALER_PATH  = MODELS_DIR / "emg_scaler.pkl"

# ─────────────────────────────────────────────
# SIGNAL ACQUISITION
# ─────────────────────────────────────────────
N_CHANNELS        = 8           # Number of EMG channels
WINDOW_SIZE       = 200         # Samples per inference window
SAMPLING_RATE     = 1000        # Hz  (adjust to your MyoWare config)
OVERLAP           = 0.5         # 50% window overlap

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
# 5 features per channel × 8 channels = 40 features
FEATURES_PER_CHANNEL = 5        # RMS, MAV, Variance, ZCR, SSC
N_FEATURES           = N_CHANNELS * FEATURES_PER_CHANNEL   # 40

ZCR_THRESHOLD = 0.0             # Threshold for Zero Crossing Rate
SSC_THRESHOLD = 0.0             # Threshold for Slope Sign Change

# ─────────────────────────────────────────────
# CLASSIFICATION
# ─────────────────────────────────────────────
CLASS_NAMES         = ["A (Nao)", "B (Sim)", "C (Talvez)"]
CLASS_LETTERS       = ["A", "B", "C"]
CLASSES             = CLASS_LETTERS          # alias kept for app.py compatibility
CLASS_COLORS        = ["#FF6B6B", "#4ECDC4", "#FFE66D"]   # per-class accent colors
N_CLASSES           = len(CLASS_NAMES)
CONFIDENCE_THRESHOLD = 0.50     # Minimum softmax probability to display a prediction

# ─────────────────────────────────────────────
# MAJORITY VOTING BUFFER
# ─────────────────────────────────────────────
VOTING_BUFFER_SIZE = 5          # Number of consecutive predictions for majority vote

# ─────────────────────────────────────────────
# INFERENCE TIMING
# ─────────────────────────────────────────────
INFERENCE_INTERVAL = 0.05       # Seconds between predictions  (20 Hz)
DISPLAY_HISTORY    = 200        # Number of raw signal samples shown on the live chart

# ─────────────────────────────────────────────
# CALIBRATION / FINE-TUNING
# ─────────────────────────────────────────────
CALIBRATION_LR      = 5e-5     # Very low learning rate to preserve learned weights
CALIBRATION_EPOCHS  = 10
CALIBRATION_BATCH   = 16
MIN_SAMPLES_PER_CLASS = 20     # Minimum samples before calibration is allowed

# ─────────────────────────────────────────────
# HARDWARE (pyserial) STUB
# ─────────────────────────────────────────────
# Replace SIMULATE_DATA with False and fill SERIAL_PORT / BAUD_RATE
# when connecting real MyoWare sensors.
SIMULATE_DATA = True
SERIAL_PORT   = "COM3"         # e.g. "COM3" on Windows, "/dev/ttyUSB0" on Linux
BAUD_RATE     = 115200

# ─────────────────────────────────────────────
# SIMULATION PARAMS
# ─────────────────────────────────────────────
SIM_NOISE_STD  = 0.05          # Base noise level for simulated signals
SIM_SIGNAL_STD = 0.2           # Amplitude of simulated burst activity