import numpy as np
import torch
from transformer_model_LOW import FullModel
import time
import matplotlib.pyplot as plt
import pandas as pd

window_length = 40000
window_overlap_percent = 50
window_overlap = int(window_length * window_overlap_percent / 100)
step = window_length - window_overlap

batch_size = 1
d_model = 8
n_heads = 2
n_layers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_range = range(3, 1000, 20)

results = []

for num_classes in class_range:
    print(num_classes)
    model = FullModel(segment_length=window_length, d_model=d_model, n_heads=n_heads,
                      n_transformer_layers=n_layers, num_classes=num_classes)
    model = model.to(device)
    model.eval()

    times = []
    for _ in range(100):

        with torch.no_grad():
                window = np.random.randn(40000).astype(np.float32)
                window_features = window.reshape(1, -1)
                window_tensor = torch.tensor(window_features, dtype=torch.float32).to(device)

                start_time = time.time()
                _ = model(window_tensor)
                elapsed = time.time() - start_time

                results.append({
                    "num_classes": num_classes,
                    "inference_time": elapsed * 1000000 / 60 / 60
                })

df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))
mean_times = df.groupby("num_classes")["inference_time"].mean()
plt.plot(mean_times.index, mean_times.values, marker='o')
plt.xlabel("Number of Classes")
plt.ylabel("Average Inference Time (s)")
plt.title("Average Transformer Inference Time by Number of Classes")
plt.tight_layout()
# plt.ylim(0, 0.25)
plt.show()

df.to_csv("inference_times.csv", index=False)

