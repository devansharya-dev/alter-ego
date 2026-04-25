# Project Alter Ego: Subvocalization EMG Detection

## 1. Project Goal (Phase 1)
The primary objective of **Project Alter Ego** is to develop a wearable system that detects subvocalized commands (words thought but not spoken) using EMG muscle signals.

**Initial Target:** Successfully detect and print the following characters:
- **A**
- **B**
- **C**
- **D**

---

## 2. Research Resources & Datasets

| Resource | Description | Use Case | Link |
| :--- | :--- | :--- | :--- |
| **Subvocalization EMG** | Dataset of 8-channel EMG recordings for subvocal speech. | Study EMG patterns and reuse CNN architecture. | [Link](https://github.com/MateusAquino/subvocalization-emg) |
| **Taskmaster** | Large-scale conversational dialogue dataset. | Future phase: Word/Sentence prediction models. | [Link](https://github.com/keithschacht/taskmaster) |
| **MyoWare Muscle Sensor** | Hardware specification and integration guide. | Real-time signal capture from neck/jaw muscles. | [Link](https://myoware.com/products/muscle-sensor/) |

---

## 3. High-Level System Flow

```
Muscle Movement (Subvocalization) 
↓ 
EMG Sensor (MyoWare) 
↓ 
Microcontroller (ESP32/Arduino) 
↓ 
Signal Preprocessing (Filtering/Normalization) 
↓ 
1D-CNN Model (Machine Learning) 
↓ 
Character Prediction 
↓ 
Output (Print A, B, C, or D)
```

---

## 4. Technical Stack

### Hardware
- **Sensors:** MyoWare Muscle Sensors (placed on throat/jaw).
- **Controller:** ESP32 (Recommended for high ADC resolution and WiFi/BT).
- **Power:** Isolated battery power (to minimize 50/60Hz noise).

### Software & ML
- **Language:** Python 3.x
- **Libraries:** 
  - `NumPy` & `SciPy` (Signal Processing)
  - `TensorFlow/Keras` (Deep Learning - CNN)
  - `PySerial` (Data streaming from hardware)
- **Model Architecture:** 1D Convolutional Neural Network (CNN) for time-series classification.

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Current)
1.  **Hardware Setup:** Wire MyoWare sensors to ESP32 and ensure "Raw EMG" signal is readable.
2.  **Data Acquisition:** 
    - Record 100 samples each for "A", "B", "C", and "D".
    - Record "Silence/Rest" samples to prevent false positives.
    - Save data as `.csv` files.
3.  **Signal Processing:** 
    - Apply Band-pass filter (20Hz - 450Hz).
    - Apply Notch filter (50Hz/60Hz).
4.  **Model Training:** Train a CNN based on the `subvocalization-emg` repository structure.
5.  **Validation:** Achieve >80% accuracy in printing the correct character in real-time.

### Phase 2: Expansion
- Increase vocabulary to the full alphabet.
- Implement word-level recognition.
- Optimize sensor placement for comfort.

### Phase 3: Intelligence
- Integrate the **Taskmaster** dataset for predictive text (Auto-complete sentences).
- Build a GUI or mobile app interface.

---

## 6. Immediate Next Steps
1. [ ] Solder MyoWare sensors and test ADC values.
2. [ ] Identify optimal electrode placement on the *Omohyoid* (neck) muscles.
3. [ ] Create a Python script to log Serial data into labeled CSV files.
4. [ ] Build the first training pipeline using TensorFlow.
