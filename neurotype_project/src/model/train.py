"""
AlterEgo – Model Fine-Tuning / Calibration (train.py)

Provides:
  • load_model_and_scaler()       – loads the pre-trained Keras model + scaler
  • calibrate_model()             – fine-tunes the last layers on new user data
  • MajorityVotingBuffer          – rolling vote buffer for stable predictions
"""

import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
from typing import Tuple, Optional

from src.config import (
    MODEL_PATH, SCALER_PATH,
    N_CLASSES, CLASS_LETTERS,
    CALIBRATION_LR, CALIBRATION_EPOCHS, CALIBRATION_BATCH,
    VOTING_BUFFER_SIZE, MIN_SAMPLES_PER_CLASS,
    N_FEATURES,
)


# ──────────────────────────────────────────────────────────────
# Model & Scaler Loading
# ──────────────────────────────────────────────────────────────

def load_model_and_scaler(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
) -> Tuple[tf.keras.Model, object]:
    """Load the pre-trained Keras model and sklearn scaler.

    Uses compile=False to avoid optimizer/format conflicts that can arise
    when loading models trained in different TF versions.

    Args:
        model_path:  Path to the .keras (or .h5) model file.
        scaler_path: Path to the pickled sklearn StandardScaler.

    Returns:
        (model, scaler) – ready for inference.

    Raises:
        FileNotFoundError: if either file is missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            "Place 'emg_final_model_abc.keras' inside models/kaggle/"
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at: {scaler_path}\n"
            "Place 'emg_scaler.pkl' inside models/kaggle/"
        )

    model = tf.keras.models.load_model(str(model_path), compile=False)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ──────────────────────────────────────────────────────────────
# Inference Helper
# ──────────────────────────────────────────────────────────────

def predict(
    model:   tf.keras.Model,
    scaler,
    features: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """Run one inference pass.

    Args:
        model:    Loaded Keras model.
        scaler:   Fitted sklearn scaler.
        features: Raw (40,) feature vector.

    Returns:
        (class_index, confidence, probabilities)
        • class_index  – integer index into CLASS_LETTERS
        • confidence   – max softmax probability  [0, 1]
        • probabilities – full softmax vector (N_CLASSES,)
    """
    x = scaler.transform(features.reshape(1, -1))          # (1, 40)
    # Model may expect (1, 40, 1) for Conv1D – reshape accordingly
    if len(model.input_shape) == 3:
        x = x.reshape(1, N_FEATURES, 1)

    probs      = model(x, training=False).numpy()[0]       # (N_CLASSES,)
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    return class_idx, confidence, probs


# ──────────────────────────────────────────────────────────────
# Majority Voting Buffer
# ──────────────────────────────────────────────────────────────

class MajorityVotingBuffer:
    """Rolling majority-vote buffer to smooth out flickering predictions.

    Keeps the last `buffer_size` predictions and returns the most
    frequent one.  Avoids the visual A⟷C or B⟷C flicker that occurs
    when consecutive softmax outputs oscillate at the decision boundary.
    """

    def __init__(self, buffer_size: int = VOTING_BUFFER_SIZE) -> None:
        self._buffer: deque = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def push(self, class_index: int) -> None:
        """Add a new raw prediction."""
        self._buffer.append(class_index)

    def vote(self) -> Optional[int]:
        """Return the majority-voted class, or None if buffer is empty."""
        if not self._buffer:
            return None
        counts = np.bincount(list(self._buffer), minlength=N_CLASSES)
        return int(np.argmax(counts))

    def push_and_vote(self, class_index: int) -> int:
        """Convenience: push + immediately return the voted class."""
        self.push(class_index)
        return self.vote()

    def reset(self) -> None:
        """Clear the buffer (e.g. on mode change)."""
        self._buffer.clear()

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.buffer_size


# ──────────────────────────────────────────────────────────────
# Calibration / Fine-Tuning
# ──────────────────────────────────────────────────────────────

def calibrate_model(
    model:     tf.keras.Model,
    scaler,
    X_raw:     np.ndarray,
    y_labels:  np.ndarray,
    lr:        float = CALIBRATION_LR,
    epochs:    int   = CALIBRATION_EPOCHS,
    batch_size: int  = CALIBRATION_BATCH,
    progress_callback=None,
) -> Tuple[tf.keras.Model, dict]:
    """Fine-tune the last dense layers of the model on new user data.

    Strategy:
    • Freeze all layers except the last two (dense + output).
    • Recompile with an Adam optimiser at a very low learning rate
      (default 5e-5) to preserve previously learned representations.
    • Optionally call progress_callback(epoch, logs) for Streamlit progress bars.

    Args:
        model:             Pre-loaded Keras model.
        scaler:            Fitted sklearn scaler (used to normalise X_raw).
        X_raw:             (n_samples, N_FEATURES) – un-scaled feature vectors.
        y_labels:          (n_samples,) – integer class indices.
        lr:                Learning rate for Adam.
        epochs:            Number of fine-tuning epochs.
        batch_size:        Mini-batch size.
        progress_callback: Optional callable(epoch: int, logs: dict) → None.

    Returns:
        (fine_tuned_model, history_dict)

    Raises:
        ValueError: if there are too few samples or not all classes are represented.
    """
    # Validation
    unique_classes = np.unique(y_labels)
    for cls in range(N_CLASSES):
        n = np.sum(y_labels == cls)
        if n < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Not enough samples for class '{CLASS_LETTERS[cls]}': "
                f"need ≥ {MIN_SAMPLES_PER_CLASS}, got {n}."
            )

    # Scale features
    X_scaled = scaler.transform(X_raw)
    if len(model.input_shape) == 3:
        X_scaled = X_scaled.reshape(-1, N_FEATURES, 1)
    y_cat = tf.keras.utils.to_categorical(y_labels, num_classes=N_CLASSES)

    # Freeze all but the last 2 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    for layer in model.layers[-2:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_log = {"loss": [], "accuracy": []}

    class _ProgressCB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            history_log["loss"].append(logs.get("loss", 0.0))
            history_log["accuracy"].append(logs.get("accuracy", 0.0))
            if progress_callback:
                progress_callback(epoch + 1, logs)

    model.fit(
        X_scaled, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[_ProgressCB()],
    )

    # Re-freeze everything for inference
    for layer in model.layers:
        layer.trainable = False

    return model, history_log


# ──────────────────────────────────────────────────────────────
# Save / Reload After Calibration
# ──────────────────────────────────────────────────────────────

def save_calibrated_model(model: tf.keras.Model, save_path: Path) -> None:
    """Persist the calibrated model back to disk."""
    model.save(str(save_path))