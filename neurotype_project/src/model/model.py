# neurotype_project/src/model/model.py
from tensorflow.keras import layers, models

def build_robust_model(input_shape=(24,)):
    """
    Kaggle wala 200 epochs optimized architecture
    """
    model = models.Sequential([
        layers.Input(shape=input_shape), 
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model