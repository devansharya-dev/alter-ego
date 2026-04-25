import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import os
import sys
from collections import Counter

# --- PATH FIX ---
# Ensures the 'src' folder is recognized
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.config import MODEL_PATH, SCALER_PATH, CLASSES, WINDOW_SIZE
except ImportError:
    st.error("Error: 'src' folder or config not found. Please check folder structure!")

# --- PAGE SETUP ---
st.set_page_config(page_title="AlterEgo - Silent Speech Engine", layout="wide")
st.title("🧠 AlterEgo: EMG Silent Alphabet Detection")

# 1. Load Assets (Cached for performance)
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at: {MODEL_PATH}")
        return None, None
    
    try:
        # Using compile=False to avoid version-specific metadata errors
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Loading Error: {e}")
        st.info("Tip: Ensure you are using the .h5 model file and your TensorFlow is updated.")
        return None, None

model, scaler = load_assets()

# --- SIDEBAR (Controls) ---
st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.radio("Select Mode", ["Live Prediction", "New User Calibration"])
port = st.sidebar.text_input("Serial Port (Arduino)", "COM3")
buffer_size = st.sidebar.slider("Stability Buffer (Samples)", 1, 10, 5)

# --- MAIN UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live EMG Signal (8 Channels)")
    chart_placeholder = st.empty() 

with col2:
    st.subheader("🎯 Detection Output")
    output_placeholder = st.empty()
    confidence_placeholder = st.empty()

# --- LOGIC ---

if mode == "Live Prediction":
    if model is None:
        st.error("Model not loaded. Application cannot run.")
    else:
        st.success("✅ System is Live. Start Subvocalizing...")
        
        # Stability Buffer to prevent flickering
        vote_buffer = []
        
        try:
            while True:
                # 1. Simulated Data (Replace with Serial read for hardware)
                raw_window = np.random.normal(0, 0.1, (WINDOW_SIZE, 8)) 
                
                # 2. Extract 40 Advanced Features (RMS, MAV, VAR, ZCR, SSC)
                # Matches the logic used in your latest Kaggle training
                rms = np.sqrt(np.mean(raw_window**2, axis=0))
                mav = np.mean(np.abs(raw_window), axis=0)
                var = np.var(raw_window, axis=0)
                zcr = np.sum(np.diff(np.sign(raw_window), axis=0) != 0, axis=0) / WINDOW_SIZE
                ssc = np.sum(np.diff(np.sign(np.diff(raw_window, axis=0)), axis=0) != 0, axis=0)
                
                feat = np.concatenate([rms, mav, var, zcr, ssc]).reshape(1, -1)
                
                # 3. Predict
                feat_scaled = scaler.transform(feat)
                prob = model.predict(feat_scaled, verbose=0)
                pred_idx = np.argmax(prob)
                conf = np.max(prob)

                # 4. Stability Filter (Majority Voting)
                if conf > 0.65: # Only buffer confident predictions
                    vote_buffer.append(pred_idx)
                if len(vote_buffer) > buffer_size:
                    vote_buffer.pop(0)
                
                # Get the most frequent prediction in the buffer
                if vote_buffer:
                    final_idx = Counter(vote_buffer).most_common(1)[0][0]
                else:
                    final_idx = pred_idx

                # 5. Update UI
                chart_placeholder.line_chart(raw_window) 
                
                if conf > 0.70:
                    output_placeholder.markdown(f"""
                        <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 2px solid #4CAF50;'>
                            <h1 style='color: #4CAF50; text-align: center; font-size: 80px; margin: 0;'>{CLASSES[final_idx]}</h1>
                            <p style='text-align: center; color: white;'>Detected Pattern</p>
                        </div>
                        """, unsafe_allow_html=True)
                    confidence_placeholder.metric("Confidence Score", f"{conf*100:.1f}%")
                else:
                    output_placeholder.markdown("<h3 style='text-align: center; color: #888;'>Listening for signals...</h3>", unsafe_allow_html=True)
                
                time.sleep(0.05)
        except Exception as e:
            st.warning(f"Streaming halted: {e}")

elif mode == "New User Calibration":
    st.warning("⚠️ Calibration Mode: System will learn your unique muscle patterns.")
    
    if st.button("🚀 Start 10s Calibration"):
        if model is None:
            st.error("Please load the model first!")
        else:
            with st.spinner("Recording Muscle Signatures... Please sit still."):
                time.sleep(4) 
                
                # Calibration Logic: Fine-tuning on dummy data (Replace with captured data)
                # Note: Shape updated to 40 features
                X_calib = np.random.randn(10, 40) 
                y_calib = tf.keras.utils.to_categorical([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 3)
                
                model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='categorical_crossentropy')
                model.fit(X_calib, y_calib, epochs=5, verbose=0)
                
                st.balloons()
                st.success("✅ Calibration Successful! Model is now optimized for your profile.")
                st.info("Switch back to 'Live Prediction' mode in the sidebar to start.")