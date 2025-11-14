import numpy as np
import torch
import csv
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

## BATCHOVE BEZ EXPORT OBRAZKU, pro všechyn varianty modelu ##

##  MULTILABELLING ##

def normalize_signal(signal):
    """Normalizuje signál."""
    signal = np.array(signal, dtype=np.float32)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val != 0:
        normalized_signal = (signal - mean_val) / std_val
    else:
        normalized_signal = signal - mean_val
    return normalized_signal



# Nastavení parametrů
# model_name = "model_20250820_170418_best"
# model_name = "model_20250820_153720_best"

# model_name = "model_20250825_100214_best"
# model_name = ["model_20250825_115512_best"]

# models_name = ["model_20250825_115512_best"]
# models_name = ["model_20250825_154418_best"]
# models_name = ["model_20250825_100214_best", "model_20250825_115512_best"]

# models_name = ["model_128x16_with_20_iter_params"]
# from transformer_model_HIGH import FullModel

# models_name = ["model_64x64_with_10_iter_params"]
# from transformer_model_MIDDLE import FullModel

# models_name = "model_8x64_d_model=8_500_epoch"
# from transformer_model_LOW import FullModel

# models_name = ["model_8x64_d_model=8_500_epoch", "model_64x64_with_10_iter_params","model_128x16_with_20_iter_params"]
models_name = ["model_128x16_with_20_iter_params"]

for model_name in models_name:

    if model_name == "model_8x64_d_model=8_500_epoch":
        from transformer_model_LOW import FullModel
        d_model = 8
    elif model_name == "model_64x64_with_10_iter_params":
        from transformer_model_MIDDLE import FullModel
        d_model = 64
    elif model_name == "model_128x16_with_20_iter_params":
        from transformer_model_HIGH import FullModel
        d_model = 64

    model_path = f"C:\\Data\\Jakubicek\\NanoGeneNetV2\\Models_final\\{model_name}.pth"
    # val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/test_5%_10G_unsegmented.csv"
    # val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/mixed_test_5%_10G_unsegmented.csv"
    # val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data\test_10G_multilabel_with_ID.csv"
    # val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data\mixed_test_5%_10G_unsegmented.csv"
    # val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data\test_10G_multilabel_with_nongene.csv"
    val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data\mixed_test_10G_fixed_multilabel_with_ID.csv"

    
    plots_dir = r"C:\Data\Jakubicek\NanoGeneNetV2\Results_Multilabel_new"

    plots_subdir = os.path.join(plots_dir, model_name)
    os.makedirs(plots_subdir, exist_ok=True)

    window_length = 40000

    for window_overlap_percent in [90,95]:  # různé překryvy [50,60,70,75,80,85,90,95]
    # for window_overlap_percent in [50,60,70,75,80,85,90,95]:
    # window_overlap_percent = 90  # překryv v procentech
        window_overlap = int(window_length * window_overlap_percent / 100)

        for batch_size in [64]:
            # for batch_size in [1, 64]:

            # d_model = 64
            n_heads = 2
            n_layers = 1
            num_classes = 11

            padding = True


            # Načtení modelu
            # device = torch.device("cpu")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            model = FullModel(segment_length=window_length, d_model=d_model, n_heads=n_heads,
                            n_transformer_layers=n_layers, num_classes=num_classes)
            model = model.to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            results_df = pd.DataFrame(columns=['Num', 'IDs', 'Predictions', 'Label', 'pos_start', 'pos_end', 'Time'])

            with open(val_csv, 'r') as infile:
                csv_reader = csv.reader(infile)

                num_rows = sum(1 for _ in csv_reader)
                infile.seek(0)

                for num in range(num_rows):
                    row = next(csv_reader)

                    # for _ in range(416):
                    #     row = next(csv_reader)
                    # row_arr = np.array([float(x) for x in row])

                    start_time = time.time()

                    # Najdi index znaku '*' v řádku
                    star_idx = None
                    for i, val in enumerate(row):
                        if val == '*':
                            star_idx = i
                            break

                    if star_idx is None:
                        raise ValueError("Znak '*' nebyl nalezen v řádku CSV.")

                    # Načti signál do pozice '*'
                    raw_signal = np.array([float(x) for x in row[:star_idx]])

                    # Za '*' jsou informace o multilabel výskytu genu
                    info_arr = row[star_idx+1:]

                    ID = info_arr[0]
                    genes_info = []
                    # Každý gen má 3 čísla: start, end, label
                    for i in range(1, len(info_arr), 3):
                        if i + 2 < len(info_arr):
                            gene_info = [info_arr[i], info_arr[i+1], info_arr[i+2]]
                            genes_info.append(gene_info)
                    num_genes = len(genes_info)

                    # Pokud je více genů, zpracuj každý zvlášť
                    signal_labels = []
                    d_starts = []
                    d_ends = []
                    for gene in genes_info:
                        d_start = int(gene[0])
                        d_end = int(gene[1])
                        signal_label = int(gene[2])

                        d_starts.append(d_start)
                        d_ends.append(d_end)
                        signal_labels.append(signal_label)

                    # Pokud není žádný gen, nastav na None
                    if not signal_labels:
                        d_starts = [None]
                        d_ends = [None]
                        signal_labels = [None]


                    signal = normalize_signal(raw_signal)


                    # Výpočet posunů okna s ošetřením okrajů a doplněním nulami
                    step = window_length - window_overlap

                    if padding:
                        # Doplnění signálu nulami na začátku i na konci
                        pad_left = window_length // 2
                        pad_right = window_length - pad_left - 1
                        signal = np.pad(signal, (pad_left, pad_right), mode='constant')
                        # Přidej pad_left ke všem d_start v d_starts
                        d_starts = [d_start + pad_left if d_start is not None else None for d_start in d_starts]
                        d_ends = [d_end + pad_left if d_end is not None else None for d_end in d_ends]

                    positions = []
                    pos = 0
                    while pos + window_length <= len(signal):
                        positions.append(pos)
                        pos += step
                    if pos < len(signal):
                        # poslední okno, které přesáhne délku signálu jen jednou
                        positions.append(len(signal) - window_length)


                    # Klasifikace v oknech
                    results = []
                    # Batch processing of windows
                    batch_windows = []
                    batch_centers = []

                    if batch_size>1:
                        for idx, pos in enumerate(positions):
                            window = signal[pos:pos + window_length]
                            batch_windows.append(window)
                            center = pos + window_length // 2  # Střed okna
                            batch_centers.append(center)

                            # If batch is full or last window
                            if len(batch_windows) == batch_size or idx == len(positions) - 1:
                                batch_features = np.stack(batch_windows)
                                batch_features = torch.tensor(batch_features, dtype=torch.float32).to(device)

                                with torch.no_grad():
                                    output = model(batch_features)
                                    klasifikace = output.argmax(dim=1).cpu().numpy()
                                    prob = torch.softmax(output, dim=1).cpu()

                                for c, k, p in zip(batch_centers, klasifikace, prob):
                                    results.append((c, k, p))

                                batch_windows = []
                                batch_centers = []
                    elif batch_size == 1:
                        for pos in positions:
                            window = signal[pos:pos + window_length]
                            window_features = window.reshape(1, -1)
                            window_features = torch.tensor(window_features, dtype=torch.float32).to(device)
                            with torch.no_grad():
                                output = model(window_features)
                            klasifikace = output.argmax(dim=1).item()
                            prob = torch.softmax(output, dim=1)
                            center = pos + window_length // 2
                            results.append((center, klasifikace, prob))
                

                    # Vykreslení klasifikace na pozice v originálním signálu
                    centers = [r[0] for r in results]
                    klasifikace = [r[1] for r in results]


                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Přidání výsledků pro aktuální signál jako nový řádek do results_df
                    # Převede množinu klasifikací na běžné int pro lepší zápis do CSV
                    unique_klasifikace = set(int(x) for x in klasifikace)
                    results_df.loc[len(results_df)] = {
                        'Num': f"{num:08d}",
                        'IDs': ID,
                        'Predictions': str(unique_klasifikace),
                        'Label': signal_labels,
                        'pos_start': d_starts,
                        'pos_end': d_ends,
                        'Time': round(elapsed_time, 8)
                    }

                # Uložení výsledků do CSV
                results_df.to_csv(os.path.join(plots_subdir, f"results_OV{window_overlap_percent}_BS{batch_size}_D{device}.csv"), index=False)