# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import numpy as np
import csv
from datetime import datetime
from transformer_dataloader import get_online_dataloader
from transformer_model_64x64 import FullModel

# Konfigurace
fold_files = [
    "/auto/brno2/home/vorochta/fold1_10G.csv",
    "/auto/brno2/home/vorochta/fold2_10G.csv",
    "/auto/brno2/home/vorochta/fold3_10G.csv",
    "/auto/brno2/home/vorochta/fold4_10G.csv",
    "/auto/brno2/home/vorochta/fold5_10G.csv"
]

# Předem nasegmentované validační soubory
validation_fold_files = [
    "/auto/brno2/home/vorochta/fold1_10G_segmented.csv",
    "/auto/brno2/home/vorochta/fold2_10G_segmented.csv",
    "/auto/brno2/home/vorochta/fold3_10G_segmented.csv",
    "/auto/brno2/home/vorochta/fold4_10G_segmented.csv",
    "/auto/brno2/home/vorochta/fold5_10G_segmented.csv"
]

# parametry
num_epochs = 250
learning_rate = 0.001
segment_length = 40000
d_model = 64
n_heads = 2
dropout = 0
n_layers = 1
num_classes = 11
val_frequency = 1  # kolikrát za epochu probíhá validace

positive_per_class = 1  # počet pozitivních segmentů na třídu v batchi
num_positive_classes = num_classes - 1  # třídy 1-10
negative_per_batch = 12  # počet negativních segmentů na batch
iterations_per_epoch = 200  # počet iterací na epochu
min_negative_padding_ratio = 0.5  # minimální poměr pro padding negativních segmentů

expected_train_batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Expected training batch size: {expected_train_batch_size}")


def merge_csv_files(file_paths, output_path):
    """Sloučí více CSV souborů do jednoho."""

    with open(output_path, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        total_rows = 0

        for file_path in file_paths:
            print(f"  Přidávám: {os.path.basename(file_path)}")
            file_rows = 0

            with open(file_path, 'r', newline='') as infile:
                csv_reader = csv.reader(infile)
                for row in csv_reader:
                    csv_writer.writerow(row)
                    file_rows += 1
                    total_rows += 1

            print(f"    Přidáno {file_rows} řádků")

    print(f"Sloučeno celkem {total_rows} řádků")
    return output_path


# Výsledky křížové validace
fold_results = []
all_fold_metrics = {
    'train_losses': [],
    'train_accuracies': [],
    'val_losses': [],
    'val_accuracies': []
}

total_cv_start_time = time.time()

print("Starting 5-Fold Cross-Validation with Single Merged Dataloader...")

for fold in range(5):
    print(f"\n{'=' * 20} FOLD {fold + 1}/5 {'=' * 20}")

    # Příprava dat pro aktuální fold
    val_file = validation_fold_files[fold]
    train_files = [fold_files[i] for i in range(5) if i != fold]

    print(f"Validation file: {os.path.basename(val_file)}")
    print(f"Training files: {[os.path.basename(f) for f in train_files]}")

    merged_train_file = f"/tmp/merged_train_fold_{fold + 1}.csv"
    merge_csv_files(train_files, merged_train_file)

    val_loader = get_online_dataloader(
        csv_file=val_file,
        batch_size=64,
        segment_length=segment_length,
        shuffle=True,
        validation=True
    )

    train_loader = get_online_dataloader(
        csv_file=merged_train_file,
        batch_size=expected_train_batch_size,
        segment_length=segment_length,
        positive_per_class=positive_per_class,
        negative_per_batch=negative_per_batch,
        iterations_per_epoch=iterations_per_epoch,
        num_classes=num_classes,
        shuffle=False,
        num_workers=0,
        validation=False,
        min_negative_padding_ratio=min_negative_padding_ratio
    )

    # Inicializace modelu pro tento fold
    model = FullModel(
        segment_length=segment_length,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        n_transformer_layers=n_layers,
        num_classes=num_classes
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Uložení nejlepšího modelu pro tento fold
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    fold_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n--- Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} ---")
        epoch_start_time = time.time()

        model.train()
        total_loss = 0.0
        total_corrects = 0
        total_samples = 0

        batch_loss = 0.0
        batch_corrects = 0
        batch_samples = 0

        total_batches = len(train_loader)

        val_interval = total_batches // val_frequency
        validation_points = []
        for i in range(val_frequency - 1):
            validation_points.append((i + 1) * val_interval)
        validation_points.append(total_batches)

        for batch_idx, batch_data in enumerate(train_loader):
            batch_segments, batch_labels = batch_data
            inputs = batch_segments.squeeze(0)  # [1, batch_size, segment_length] -> [batch_size, segment_length]
            labels = batch_labels.squeeze(0)  # [1, batch_size] -> [batch_size]

            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item() * inputs.size(0)
            batch_corrects += torch.sum(preds == labels).item()
            batch_samples += inputs.size(0)

            total_loss += loss.item() * inputs.size(0)
            total_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

            if (batch_idx + 1) in validation_points:
                part_train_loss = batch_loss / batch_samples if batch_samples > 0 else 0
                part_train_acc = batch_corrects / batch_samples * 100 if batch_samples > 0 else 0
                print(
                    f"[Batch {batch_idx + 1}/{total_batches}] Train Loss: {part_train_loss:.4f} Acc: {part_train_acc:.2f}%")

                # Validace
                model.eval()
                val_loss = 0.0
                val_corrects = 0
                val_samples = 0
                with torch.no_grad():
                    for val_batch_data in val_loader:
                        val_inputs, val_labels = val_batch_data
                        val_inputs = val_inputs.to(device, dtype=torch.float)
                        val_labels = val_labels.to(device, dtype=torch.long)
                        val_outputs = model(val_inputs)
                        v_loss = criterion(val_outputs, val_labels)
                        v_preds = torch.argmax(val_outputs, dim=1)
                        val_loss += v_loss.item() * val_inputs.size(0)
                        val_corrects += torch.sum(v_preds == val_labels).item()
                        val_samples += val_inputs.size(0)

                val_epoch_loss = val_loss / val_samples
                val_epoch_acc = val_corrects / val_samples * 100
                print(f"[Batch {batch_idx + 1}] Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.2f}%")

                # Uložení nejlepšího modelu pro tento fold
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("Saving best model for this fold...")

                model.train()
                batch_loss = 0.0
                batch_corrects = 0
                batch_samples = 0

        epoch_train_loss = total_loss / total_samples
        epoch_train_acc = total_corrects / total_samples * 100
        print(f"[End of Epoch] Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}%")

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_samples = 0
        with torch.no_grad():
            for val_batch_data in val_loader:
                val_inputs, val_labels = val_batch_data
                val_inputs = val_inputs.to(device, dtype=torch.float)
                val_labels = val_labels.to(device, dtype=torch.long)
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                v_preds = torch.argmax(val_outputs, dim=1)
                val_loss += v_loss.item() * val_inputs.size(0)
                val_corrects += torch.sum(v_preds == val_labels).item()
                val_samples += val_inputs.size(0)

        final_val_loss = val_loss / val_samples
        final_val_acc = val_corrects / val_samples * 100
        print(f"[End of Epoch] Validation Loss: {final_val_loss:.4f} Acc: {final_val_acc:.2f}%")

        epoch_time = (time.time() - epoch_start_time) / 60
        print(f"Epoch {epoch + 1} finished in {epoch_time:.2f} minutes")

        all_fold_metrics['train_losses'].append(epoch_train_loss)
        all_fold_metrics['train_accuracies'].append(epoch_train_acc)
        all_fold_metrics['val_losses'].append(final_val_loss)
        all_fold_metrics['val_accuracies'].append(final_val_acc)

    fold_time = (time.time() - fold_start_time) / 60
    print(f"\nFold {fold + 1} finished in {fold_time:.2f} minutes")
    print(f"Best validation loss for fold {fold + 1}: {best_val_loss:.4f}")

    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.load_state_dict(best_model_wts)
    model_path = f"/auto/brno2/home/vorochta/model_fold_{fold + 1}_{unique_id}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Best model for fold {fold + 1} saved as {os.path.basename(model_path)}")

    if os.path.exists(merged_train_file):
        os.remove(merged_train_file)
        print(f"Removed temporary file: {os.path.basename(merged_train_file)}")

    fold_results.append({
        'fold': fold + 1,
        'best_val_loss': best_val_loss,
        'final_val_acc': final_val_acc,
        'final_train_loss': epoch_train_loss,
        'final_train_acc': epoch_train_acc,
        'model_path': model_path
    })

total_cv_time = (time.time() - total_cv_start_time) / 60
print("5-FOLD CROSS-VALIDATION RESULTS")
print(f"Total cross-validation time: {total_cv_time:.2f} minutes")

# Výpis výsledků pro každý fold
for result in fold_results:
    print(f"\nFold {result['fold']}:")
    print(f"  Best Validation Loss: {result['best_val_loss']:.4f}")
    print(f"  Final Validation Accuracy: {result['final_val_acc']:.2f}%")
    print(f"  Final Training Loss: {result['final_train_loss']:.4f}")
    print(f"  Final Training Accuracy: {result['final_train_acc']:.2f}%")
    print(f"  Model saved: {os.path.basename(result['model_path'])}")

# Statistiky
val_losses = [result['best_val_loss'] for result in fold_results]
val_accuracies = [result['final_val_acc'] for result in fold_results]

print(f"\n{'=' * 60}")
print("CROSS-VALIDATION STATISTICS")
print(f"{'=' * 60}")
print(f"Mean Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
print(f"Mean Validation Accuracy: {np.mean(val_accuracies):.2f}% ± {np.std(val_accuracies):.2f}%")
print(f"Best Fold (lowest val loss): Fold {np.argmin(val_losses) + 1} (Loss: {np.min(val_losses):.4f})")
print(f"Best Fold (highest val acc): Fold {np.argmax(val_accuracies) + 1} (Acc: {np.max(val_accuracies):.2f}%)")

# Uložení výsledků
results_file = f"/auto/brno2/home/vorochta/crossval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(results_file, 'w') as f:
    f.write("5-FOLD CROSS-VALIDATION RESULTS WITH MERGED DATALOADER\n")
    f.write("=" * 60 + "\n")
    f.write(f"Total cross-validation time: {total_cv_time:.2f} minutes\n\n")

    f.write("CONFIGURATION:\n")
    f.write(f"  Training files: {[os.path.basename(f) for f in fold_files]}\n")
    f.write(f"  Validation files: {[os.path.basename(f) for f in validation_fold_files]}\n")
    f.write("DATALOADER CONFIGURATION:\n")
    f.write(f"  Positive per class: {positive_per_class}\n")
    f.write(f"  Negative per batch: {negative_per_batch}\n")
    f.write(f"  Iterations per epoch: {iterations_per_epoch}\n")
    f.write(f"  Expected batch size: {expected_train_batch_size}\n")
    f.write(f"  Segment length: {segment_length}\n\n")

    for result in fold_results:
        f.write(f"Fold {result['fold']}:\n")
        f.write(f"  Best Validation Loss: {result['best_val_loss']:.4f}\n")
        f.write(f"  Final Validation Accuracy: {result['final_val_acc']:.2f}%\n")
        f.write(f"  Final Training Loss: {result['final_train_loss']:.4f}\n")
        f.write(f"  Final Training Accuracy: {result['final_train_acc']:.2f}%\n")
        f.write(f"  Model saved: {os.path.basename(result['model_path'])}\n\n")

    f.write("CROSS-VALIDATION STATISTICS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Mean Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}\n")
    f.write(f"Mean Validation Accuracy: {np.mean(val_accuracies):.2f}% ± {np.std(val_accuracies):.2f}%\n")
    f.write(f"Best Fold (lowest val loss): Fold {np.argmin(val_losses) + 1} (Loss: {np.min(val_losses):.4f})\n")
    f.write(f"Best Fold (highest val acc): Fold {np.argmax(val_accuracies) + 1} (Acc: {np.max(val_accuracies):.2f}%)\n")

print(f"\nResults saved to: {os.path.basename(results_file)}")
print("Cross-validation completed!")