# neurotype_project/emg_pipeline.py
from src.model.train import train_base_model
from src.model.evaluate import evaluate_and_plot
# Yahan data loading logic aayega (jo Kaggle pe kiya tha)

def run_full_pipeline():
    # 1. Load Data
    print("Loading data...")
    # X_train, y_train, X_test, y_test = load_and_preprocess_everything()
    
    # 2. Train
    # model = train_base_model(X_train, y_train)
    
    # 3. Evaluate
    # evaluate_and_plot(model, X_test, y_test)
    pass

if __name__ == "__main__":
    run_full_pipeline()