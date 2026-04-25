# evaluate.py
# neurotype_project/src/model/evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASSES

def evaluate_and_plot(model, X_test, y_test):
    """
    Model performance visualization
    """
    # 1. Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 2. Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # Save plot to outputs/plots
    plt.savefig("outputs/plots/confusion_matrix.png")
    print("✅ Confusion Matrix saved to outputs/plots/")