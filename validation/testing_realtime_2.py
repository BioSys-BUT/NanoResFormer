import numpy as np
import torch
import csv
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

## BATCHOVE BEZ EXPORT OBRAZKU, pro všechyn varianty modelu ##

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

models_name = ["model_64x64_with_10_iter_params"]
# models_name = ["model_8x64_with_20_iter_params"]
# models_name = ["model_128x16_with_20_iter_params"]


# from transformer_model_MIDDLE import FullModel
from transformer_model_HIGH import FullModel

# models_name = ["model_20250825_100214_best", "model_20250825_115512_best"]

# device = torch.device("cpu")
# device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for device_type in ["cuda"]:
        
    for model_name in models_name:

        # model_path = f"C:\\Data\\Jakubicek\\NanoGeneNetV2\\Models\\{model_name}.pth"
        model_path = f"C:\\Data\\Jakubicek\\NanoGeneNetV2\\Models_final\\{model_name}.pth"
        # val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/test_5%_10G_unsegmented.csv"
        val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/mixed_test_5%_10G_unsegmented.csv"


        plots_dir = r"C:\Data\Jakubicek\NanoGeneNetV2\Results_Acc"

        plots_subdir = os.path.join(plots_dir, model_name)
        os.makedirs(plots_subdir, exist_ok=True)

        device = torch.device(device_type)

        d_model = 64
        n_heads = 2
        n_layers = 1
        num_classes = 11
        window_length = 40000
        
        model = FullModel(segment_length=window_length, d_model=d_model, n_heads=n_heads,
                    n_transformer_layers=n_layers, num_classes=num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()



        for window_overlap_percent in [50,60,70,75,80,85,90,95]:  # různé překryvy
        # window_overlap_percent = 90  # překryv v procentech
            window_overlap = int(window_length * window_overlap_percent / 100)

            for batch_size in [64]:
                # for batch_size in [1, 64]:

                padding = True

                # results_df = pd.DataFrame(columns=['num', 'center', 'klasifikace', 'probabilities','unique_klasifikace']) 
                results_df = pd.DataFrame(columns=['Num', 'Predictions','Label','Time'])

                start_time_all = time.time()

                # Načtení třetího signálu (třetí řádek)
                with open(val_csv, 'r') as infile:
                    csv_reader = csv.reader(infile)

                    num_rows = sum(1 for _ in csv_reader)
                    infile.seek(0)

                    for num in range(num_rows):
                        row = next(csv_reader)
                        row_arr = np.array([float(x) for x in row])


                    # for _ in range(800):
                    #     row = next(csv_reader)
                    # row_arr = np.array([float(x) for x in row])

                        start_time = time.time()

                        d_start = int(row_arr[-3])
                        d_end = int(row_arr[-2])
                        signal_label = int(row_arr[-1])
                        raw_signal = row_arr[:-3]

                        signal = normalize_signal(raw_signal)


                        # Výpočet posunů okna s ošetřením okrajů a doplněním nulami
                        step = window_length - window_overlap

                        if padding:
                            # Doplnění signálu nulami na začátku i na konci
                            pad_left = window_length // 2
                            pad_right = window_length - pad_left - 1
                            signal = np.pad(signal, (pad_left, pad_right), mode='constant')
                            d_start += pad_left
                            d_end += pad_left

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
                            'Predictions': str(unique_klasifikace),
                            'Label': signal_label,
                            'Time': round(elapsed_time, 8)
                        }

                end_time_all = time.time()

                elapsed_time_all = end_time_all - start_time_all
                Time_all = (elapsed_time_all / num_rows) * 1_000_000 / 3600
                print(f"Celkový čas zpracování pro {num} signálů: {elapsed_time_all/60:.2f} minut")
                print(f"Průměrný čas na 1 milion signálů: {Time_all:.2f} hodin")

                # Uložení výsledků do CSV
                results_df.to_csv(os.path.join(plots_subdir, f"results_OV{window_overlap_percent}_BS{batch_size}_D{device}_T{Time_all:.2f}.csv"), index=False)

