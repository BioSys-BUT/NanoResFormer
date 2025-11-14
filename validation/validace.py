import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
from transformer_dataloader import get_online_dataloader
# from transformer_model_LOW import FullModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score

# model_name = "model_64x64_with_10_iter_params"
# model_name = "model_8x64_d_model=8_500_epoch"
model_name = "model_128x16_with_20_iter_params"

if model_name == "model_8x64_d_model=8_500_epoch":
    from transformer_model_LOW import FullModel
    d_model = 8
elif model_name == "model_64x64_with_10_iter_params":
    from transformer_model_MIDDLE import FullModel
    d_model = 64
elif model_name == "model_128x16_with_20_iter_params":
    from transformer_model_HIGH import FullModel
    d_model = 64


model_path = r"C:\Data\Jakubicek\NanoGeneNetV2\Models_final"
val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/valid_15%_10G_segmented.csv"
batch_size = 64
segment_length = 40000
# d_model = 64
n_heads = 2
dropout = 0.0
n_layers = 1
num_classes = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FullModel(segment_length=segment_length, d_model=d_model, n_heads=n_heads,
                  dropout=dropout, n_transformer_layers=n_layers, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(f"{model_path}\\{model_name}.pth"))
model.eval()

val_loader = get_online_dataloader(
    csv_file=val_csv,
    batch_size=batch_size,
    segment_length=segment_length,
    shuffle=True,
    validation=True,
    num_workers=0,
    num_classes=num_classes
)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    model.eval()
    for val_batch_data in val_loader:
        val_batch_segments, val_batch_labels = val_batch_data
        val_inputs = val_batch_segments.squeeze(0)
        val_labels = val_batch_labels.squeeze(0)

        val_inputs = val_inputs.to(device, dtype=torch.float)
        val_labels = val_labels.to(device, dtype=torch.long)
        val_outputs = model(val_inputs)
        v_preds = torch.argmax(val_outputs, dim=1)

        all_preds.extend(v_preds.cpu().numpy())
        all_labels.extend(val_labels.cpu().numpy())
        all_probs.append(torch.softmax(val_outputs, dim=1).cpu().numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

cm = confusion_matrix(all_labels, all_preds)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

gene_names = [
    "no gene", "blaSHV", "blaOXA", "aac(3)", "aph(6)-Id", "aph(3'')-Ib",
    "OqxA", "OqxB", "tetA", "tetD", "fosA"
]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=gene_names, yticklabels=gene_names)
plt.title("Confusion Matrix (%)")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(rotation=45)
plt.savefig(f"{model_path}\\CM_{model_name}.png")
plt.close()
# plt.show()

# Binary confusion matrix for gene (>0) vs non-gene (0) signals
binary_labels = (all_labels > 0).astype(int)
binary_preds = (all_preds > 0).astype(int)
cm_binary = confusion_matrix(binary_labels, binary_preds)
cm_binary_percent = cm_binary.astype('float') / cm_binary.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(5, 4))
sns.heatmap(cm_binary_percent, annot=True, fmt=".1f", cmap="Greens",
            xticklabels=["Non-gene", "Gene"], yticklabels=["Non-gene", "Gene"])
plt.title("Binary Confusion Matrix (%)")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(rotation=45)
# plt.yticks(rotation=45)
plt.savefig(f"{model_path}\\Binary_CM_{model_name}.png")
plt.close()

# ROC AUC analysis for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    # Binarize labels for class i
    y_true = (all_labels == i).astype(int)
    y_score = all_probs[:, i]
    fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{gene_names[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Classes")
plt.legend(loc="lower right")
plt.savefig(f"{model_path}\\ROC_{model_name}.png")
plt.close()
# plt.show()

precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
overall_accuracy = (all_labels == all_preds).mean()

with open(f"{model_path}\\metrics_{model_name}.txt", "w") as f:
    for i in range(num_classes):
        f.write(f"{gene_names[i]}: Precision = {precision_per_class[i]:.3f}, Sensitivity (Recall) = {recall_per_class[i]:.3f}\n")
    f.write(f"\nOverall Accuracy: {overall_accuracy:.3f}\n")





