# AlterEgo Project Structure

Here is a detailed breakdown of the project architecture. Each file and directory has a specific role in managing data, training models, and deploying the app.

*(Note: The actual `data` folder hierarchy has been omitted from this tree as requested.)*

```text
D:\altter ego
|   app.py                           # 🚀 Main Streamlit web application for Live Prediction & UI
|   emg_pipeline.py                  # ⚙️ Entry point for running the complete training pipeline
|   convert_data.py                  # 🔄 Script to handle external dataset format conversions
|   emg_model.h5                     # 🧠 Legacy root saved trained model
|   PROJECT_PLAN.md                  # 📝 Project roadmap and task tracking
|   Project_Report_KeyPoints.md      # 📑 Key bullet points for project documentation/report
|   EMG Silent Alphabet Detection System (1).docx # 📘 Complete project thesis/report
|
\---neurotype_project                # 📁 Main Project Module Directory
    |   README.md                    # 📖 Project documentation and setup instructions 
    |   requirements.txt             # 📦 Python package dependencies
    |
    +---models                       # 🤖 Saved Models & Scalers
    |   +---kaggle
    |   |       emg_final_model_abc.keras # ✔️ Highly optimized pre-trained model
    |   |       emg_scaler.pkl            # ⚖️ Scaler for feature normalization
    |   +---saved                    # 💾 Directory for user fine-tuned models
    |   \---scaler                   # 📈 Additional scalers directory
    |
    +---notebooks                    # 📓 Data Science Notebooks
    |       exploration.ipynb        # 📊 Exploratory Data Analysis (EDA) on EMG signals
    |       training.ipynb           # 🏋️ Model architecture experimentation & prototyping
    |
    +---outputs                      # 📈 Logs & Visualizations
    |   +---logs                     # 📜 TensorBoard or training loss/accuracy logs
    |   \---plots                    # 📊 Exported evaluation charts (e.g., Confusion Matrix)
    |
    \---src                          # 💻 Core Source Code
        |   config.py                # 🛠️ Central configuration repository (Hyperparams, Paths)
        |   helpers.py               # 🧩 Reusable helper and utility functions
        |
        +---model                    # 🏗️ ML Model definitions and processes
        |       model.py             # 📐 Model architecture definition (Layers, Dropouts)
        |       train.py             # 🏃 Base model training & user fine-tuning logic
        |       evaluate.py          # 📉 Testing evaluation, metric calculations & plotting
        |
        \---preprocessing            # 🔬 Feature Engineering & Processing
                filters.py           # 🧹 Noise reduction and raw signal filtering logic
                windowing.py         # 🪟 Feature extraction (RMS, MAV, VAR) using sliding windows
                normalize.py         # 📏 Standardization algorithms for raw EMG
```
