import numpy as np
import torch
import csv
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

## pro vizualizaci signálu, s exportem obrázků pro každý signál, jen pro jeden model

## ale pro nove signaly MULTILABEL ##


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
# model_name = "model_20250825_154418_best"

# model_name = "model_128x16_with_20_iter_params"
# model_name = "model_8x64_d_model=8_500_epoch"
model_name = "model_64x64_with_10_iter_params"



model_path = f"C:\\Data\\Jakubicek\\NanoGeneNetV2\\Models_final\\{model_name}.pth"
# val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data/test_10G_multilabel_with_ID.csv"
val_csv = r"C:\Data\Jakubicek\NanoGeneNetV2\Data\transformer_nanopore_data\mixed_test_10G_fixed_multilabel_with_ID.csv"


plots_dir = r"C:\Data\Jakubicek\NanoGeneNetV2\Results_Multilabel_new\images_selected"


plots_subdir = os.path.join(plots_dir, model_name)
os.makedirs(plots_subdir, exist_ok=True)

window_length = 40000
window_overlap_percent = 90  # překryv v procentech
window_overlap = int(window_length * window_overlap_percent / 100)

batch_size = 1
d_model = 8
n_heads = 2
n_layers = 1
num_classes = 11

padding = True


# Načtení modelu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_name == "model_8x64_d_model=8_500_epoch":
    from transformer_model_LOW import FullModel
    d_model = 8
elif model_name == "model_64x64_with_10_iter_params":
    from transformer_model_MIDDLE import FullModel
    d_model = 64
elif model_name == "model_128x16_with_20_iter_params":
    from transformer_model_HIGH import FullModel
    d_model = 64

model = FullModel(segment_length=window_length, d_model=d_model, n_heads=n_heads,
                  n_transformer_layers=n_layers, num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# results_df = pd.DataFrame(columns=['num', 'center', 'klasifikace', 'probabilities','unique_klasifikace']) 
results_df = pd.DataFrame(columns=['num', 'unique_klasifikace','Time'])

# num_sig = [175, 307, 539, 700, 1854]  # čísla signálů k vizualizaci (0-indexováno)
num_sig = [175,]


# Načtení třetího signálu (třetí řádek)
with open(val_csv, 'r') as infile:
    csv_reader = csv.reader(infile)

    num_rows = sum(1 for _ in csv_reader)
    infile.seek(0)

    for num in range(num_rows):
        row = next(csv_reader)

        if num not in num_sig:
            continue

    #     row_arr = np.array([float(x) for x in row])

    # for num in range(num_rows):
    # for num in range(num_sig, num_sig+1):
    #     row = next(csv_reader)

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
        print(ID)
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


        start_time = time.time()

        signal = normalize_signal(raw_signal)

        # Výpočet posunů okna s ošetřením okrajů a doplněním nulami
        step = window_length - window_overlap

        if padding:
            # Doplnění signálu nulami na začátku i na konci
            pad_left = window_length // 2
            pad_right = window_length - pad_left - 1
            signal = np.pad(signal, (pad_left, pad_right), mode='constant')
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
        for pos in positions:
            window = signal[pos:pos + window_length]
            window_features = window.reshape(1, -1)
            window_features = torch.tensor(window_features, dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(window_features)
            # output = model(window_features)
            klasifikace = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)
            center = pos + window_length // 2  # Střed okna
            # center = pos  # Střed okna s ohledem na padding
            results.append((center, klasifikace, prob))
            

        # Vykreslení klasifikace na pozice v originálním signálu
        centers = [r[0] for r in results]
        klasifikace = [r[1] for r in results]

        # Přidání výsledků pro aktuální signál jako nový řádek do results_df
        results_df.loc[len(results_df)] = {
            'num': f"{num:08d}",
            # 'center': centers,
            # 'klasifikace': klasifikace,
            'unique_klasifikace': set(klasifikace),
            'správný label': signal_label
        }

        end_time = time.time()
        elapsed_time = end_time - start_time
        results_df.loc[len(results_df) - 1, 'Time'] = round(elapsed_time, 4)

        # plt.figure(figsize=(15, 5))
        # plt.plot(range(len(signal)), signal, label='Signal', alpha=0.7)
        # plt.bar(centers, klasifikace, color='red', label='Classification', width=window_length, alpha=0.5)
        # plt.scatter(centers, klasifikace, color='black', label='Window Center', s=10)

        # # Zvýraznění oblasti genů
        # # plt.axvspan(d_start + pad_left, d_end + pad_left, color='green', alpha=0.3, label='Gene Region')
        # plt.axvspan(d_start, d_end, color='green', alpha=0.3, label='Gene Region')

        # plt.xlabel('Position in Signal')
        # plt.ylabel('Signal / Classification')
        # plt.legend()
        # plt.title(f"Predikovaný labely: {set(klasifikace)}, Správný label: {signal_label}")
        # plt.show()

        # Zkrácení signálu zpět na původní délku (odstranění paddingu)
        if padding:
            signal = signal[pad_left:-pad_right if pad_right > 0 else None]
            centers = [c - pad_left for c in centers]
            d_starts = [d_start - pad_left if d_start is not None else None for d_start in d_starts]
            d_ends = [d_end - pad_left if d_end is not None else None for d_end in d_ends]


        # Nastav kontrastní barvy pro geny 2, 3 a 8
        label_colors = [None, "#e6194B", "#0e3ce4", "#f58231", "#911eb4", "#46f0f0",  "#009e15",
                "#f032e6", "#bcf60c", "#5E4219", "#008080"]
        
        gene_names = [
            "no gene", "blaSHV", "blaOXA", "aac(3)", "aph(6)-Id", "aph(3'')-Ib",
            "OqxA", "OqxB", "tetA", "tetD", "fosA"
            ]

        
        plt.figure(figsize=(15, 5))
        ax1 = plt.gca()
        signal_plot, = ax1.plot(range(len(signal)), signal, label='Signal', alpha=0.5)
        ax1.set_xlabel('Position in signal')
        ax1.set_ylabel('Standardized signal (z-score)')
        ax1.set_ylim(-3.0, 3.5)
        ax1.set_xlim(0, len(signal))

        # Zvýrazni všechny oblasti genů s barvou podle labelu, kromě labelu 0
        gene_region_handles = []
        gene_region_labels = []
        for d_start, d_end, label in zip(d_starts, d_ends, signal_labels):
            if d_start is not None and d_end is not None and label is not None and label != 0:
                handle = ax1.axvspan(d_start, d_end, color=label_colors[label], alpha=0.3)
                gene_region_handles.append(handle)
                gene_region_labels.append(f'{gene_names[label]}')

        ax2 = ax1.twinx()
        prob_handles = []
        prob_labels = []
        # Zobraz pravděpodobnosti pro každou třídu s odpovídající barvou, kromě třídy 0
        for class_idx in range(1, num_classes):
            prob_values = [r[2][0, class_idx].item() for r in results]
            handle, = ax2.plot(centers, prob_values, label=f'{gene_names[class_idx]}', alpha=0.7, color=label_colors[class_idx])
            prob_handles.append(handle)
            prob_labels.append(f'{gene_names[class_idx]}')

        ax2.set_ylabel('Detection Probability')
        ax2.set_yticks([0.0, 1.0])
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_xlim(0, len(signal))

        # Přidej tenkou vodorovnou přerušovanou linku na hodnotu 1 probability
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0, alpha=1.0)

        # Legend 1: Signal
        legend1 = ax1.legend([signal_plot], ['Signal'], loc='upper left', bbox_to_anchor=(0.0, 1.0))
        ax1.add_artist(legend1)

        # Legend 2: Gene regions (všechny možné geny, i když nejsou v signálu)
        legend2_handles = []
        legend2_labels = []
        for label in range(1, num_classes):
            legend2_handles.append(plt.Line2D([0], [0], color=label_colors[label], lw=8, alpha=0.3))
            legend2_labels.append(f'{gene_names[label]}')
        legend2 = ax1.legend(legend2_handles, legend2_labels, loc='upper left', bbox_to_anchor=(0.0, 1.2), ncol=5)
        ax1.add_artist(legend2)

        # Legend 3: Probabilities
        legend3 = ax2.legend(prob_handles, prob_labels, loc='upper left', bbox_to_anchor=(0.5, 1.2), ncol=5)
        ax2.add_artist(legend3)

        # Probability for "no gene" (class 0)
        prob_no_gene = [r[2][0, 0].item() for r in results]
        handle_no_gene, = ax2.plot(centers, prob_no_gene, label='no gene', color='gray', linestyle='--', alpha=0.8)

        # Legend 4: "no gene" probability
        legend4 = ax2.legend([handle_no_gene], ['no gene'], loc='upper right', bbox_to_anchor=(1.0, 1.0))
        ax2.add_artist(legend4)
        ax2.add_artist(legend4)


        # plt.subplots_adjust(top=0.7)
        # # plt.savefig(os.path.join(plots_subdir, f"signal_{num:08d}.png"))
        # plt.savefig(os.path.join(plots_subdir, f"signal_{num:08d}_2.png"))
        # plt.close()


# # Uložení výsledků do CSV
# results_df.to_csv(os.path.join(plots_subdir, "results.csv"), index=False)