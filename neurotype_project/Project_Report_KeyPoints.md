# EMG-Based Silent Speech Classification: Project Summary & Insights

## 1. Project Goal
Building a machine learning pipeline to classify Surface Electromyography (EMG) signals into three intent classes: **Nao (No)**, **Sim (Yes)**, and **Talvez (Maybe)**.

## 2. The Core Challenge: Physiological Variance
The primary hurdle in EMG-based systems is that **EMG signals are highly subject-dependent**. 
* **Factors:** Muscle mass, skin impedance, sensor placement, and individual speech patterns differ significantly from person to person.
* **Observation:** A model trained on *User A* will naturally perform poorly on *User B* (~20-30% accuracy) because it perceives the new user's muscle signature as "noise."

## 3. Our Solution: Calibration-Based Transfer Learning
To overcome the variance, we implemented a **two-stage training pipeline**:

### Stage 1: Base Model (Feature Extraction)
* Trained on a diverse pool of 7-8 users.
* **Goal:** To learn the general "language" of muscle contractions.
* **Architecture:** 1D-CNN or Dense MLP using statistical features (**RMS, MAV, Variance**).

### Stage 2: User-Specific Calibration (Fine-Tuning)
* When a new user wears the device, a **10-second calibration** is performed.
* We use a small fraction (e.g., 20%) of the new user's data to "fine-tune" the pre-trained model.
* **Result:** This shifts the model's decision boundaries to align with the new user's unique physiology, jumping accuracy from **~27% to 95%+**.

## 4. Signal Processing Highlights
* **Z-Score Normalization (Per-File):** Essential to remove hardware-induced offsets (like the -187500.0 baseline shifts).
* **Feature Engineering:** Instead of raw noisy signals, we extract **Root Mean Square (RMS)** and **Mean Absolute Value (MAV)** to capture the energy envelope of the muscle movement.
* **Majority Voting:** Implemented a prediction buffer to ensure the output remains stable even if the hardware produces momentary noise spikes.

## 5. Technical Impact
This approach transforms the system from a generic classifier into an **Adaptive AI System**, making it viable for real-world medical or assistive technology (like Smart Glasses for the speech-impaired).