import os
import pandas as pd
import numpy as np

base_path = r"D:\altter ego\neurotype_project\data\saves"

mapping = {
    "Nao": "A",
    "Sim": "B",
    "Talvez": "C"
}

all_data = []
all_labels = []

for person in os.listdir(base_path):
    person_path = os.path.join(base_path, person)

    if not os.path.isdir(person_path):
        continue

    for condition in ["A", "F"]:
        cond_path = os.path.join(person_path, condition)

        if not os.path.exists(cond_path):
            continue

        for file in os.listdir(cond_path):
            if file.endswith(".csv"):
                file_path = os.path.join(cond_path, file)

                df = pd.read_csv(file_path)

                label_key = file.split(".")[0]  # Nao, Sim, Talvez
                label = mapping.get(label_key, None)

                if label is None:
                    continue

                channels = [f"Channel_{i}" for i in range(1, 9)]
                data = df[channels].values

                labels = [label] * len(data)

                all_data.append(data)
                all_labels.extend(labels)

# combine all
X_raw = np.vstack(all_data)
y_raw = np.array(all_labels)

print("Total samples:", X_raw.shape)
print("Labels:", np.unique(y_raw))


window_size = 100
stride = 50

X = []
y = []

for i in range(0, len(X_raw) - window_size, stride):
    window = X_raw[i:i+window_size]
    label_window = y_raw[i:i+window_size]

    # majority label
    vals, counts = np.unique(label_window, return_counts=True)
    majority = vals[np.argmax(counts)]

    X.append(window)
    y.append(majority)

X = np.array(X)
y = np.array(y)

print("Windowed X shape:", X.shape)
print("Windowed y shape:", y.shape)
print("Labels:", np.unique(y))