import os
import csv
import numpy as np
import pandas as pd
import torch
from utils import normalize_signal, determine_batch_size, pad_signal, depad_signal
from utils import prepare_windows, select_device, export_image, model_mapping
import sys

def NanoResFormer(csv_path,
                  out_dir,
                  csv_name=None,
                  OV=80,
                  Model='middle',
                  export_images=False,
                  device_pref='auto'):
    """
    Run NanoResFormer inference on signals listed in a CSV file.

    Parameters:
        csv_path (str): Path to input CSV file containing raw signals and optional metadata.
                        Each row should contain signal samples followed by '*' and optional info.
        out_dir (str): Directory where results (CSV and optional images) will be saved.
        csv_name (str, optional): Base name for output files; if None the input filename (without
                                  extension) will be used. Default: None.
        OV (int or float, optional): Window overlap percentage (0-99). Determines overlap between
                                     consecutive sliding windows. Default: 80.
        Model (str, optional): Model variant to use (e.g., 'middle', 'small', 'big'). Default: 'middle'.
        export_images (bool, optional): If True, export annotated images for each processed signal.
                                        Default: False.
        device_pref (str, optional): Device selection preference: 'cpu', 'cuda', or 'auto' to let the
                                     code choose. Default: 'auto'.

    Returns:
        str: Path to the CSV file containing aggregated results for all processed signals.
    """

    # Print software / license / citation info
    print("----------------------------------------")
    print("Starting NanoResFormer - Nanopore Signal Transformer for Gene Detection")
    print("Version: 2.0")
    print("Licence: MIT (see project LICENSE file)")
    print("Please cite: Jakubicek et al., NanoResFormer (2025)")
    print("----------------------------------------")

    print("Starting NanoResFormer inference...")


    # Print input filename (with full path) and output directory
    print(f"Input CSV path: {os.path.abspath(csv_path)}")
    print(f"Output directory: {os.path.abspath(out_dir)}")

    # warn if provided csv_name matches input csv filename (without extension)
    input_base = os.path.splitext(os.path.basename(csv_path))[0]
    if csv_name:
        provided_base = os.path.splitext(csv_name)[0]
        if provided_base == input_base:
            print("---Warning--- Provided csv_name matches input CSV filename without extension. This may overwrite the original file if out_dir points to the same directory as the input CSV.")


    # Quick scan: count signals (rows containing '*') and print the count
    print("Counting signals in the input CSV file...")
    total_signals = 0
    with open(csv_path, 'r', newline='') as _in:
        csv_reader = csv.reader(_in)
        for row in csv_reader:
            if '*' in row:
                total_signals += 1
    print(f"Number of signals found in file: {total_signals}")

    window_length = 40000
    n_heads = 2
    n_layers = 1
    num_classes = 11
    padding = True

    batch_size = determine_batch_size(window_length, device_pref)

    os.makedirs(out_dir, exist_ok=True)
    csv_name = csv_name or os.path.splitext(os.path.basename(csv_path))[0]+"_results"

    images_dir = os.path.join(out_dir, f"images_{csv_name}")
    if export_images:
        os.makedirs(images_dir, exist_ok=True)

    FullModelClass, model_name, d_model = model_mapping(Model)
    model_path = f"C:\\Data\\Jakubicek\\NanoGeneNetV2\\Models_final\\{model_name}.pth"

    device = select_device(device_pref)

    model = FullModelClass(segment_length=window_length, d_model=d_model, n_heads=n_heads,
                           n_transformer_layers=n_layers, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    window_overlap_percent = OV
    window_overlap = int(window_length * window_overlap_percent / 100)
    step = int(window_length - window_overlap)
    if step <= 0:
        raise ValueError("OV too large, resulting in non-positive step size.")

    # no Time column as requested
    results_df = pd.DataFrame(columns=['num', 'ID', 'found_genes'])

    label_colors = [None, "#e6194B", "#0e3ce4", "#f58231", "#911eb4", "#46f0f0", "#009e15",
                    "#f032e6", "#bcf60c", "#5E4219", "#008080"]
    gene_names = [
        "no gene", "blaSHV", "blaOXA", "aac(3)", "aph(6)-Id", "aph(3'')-Ib",
        "OqxA", "OqxB", "tetA", "tetD", "fosA"
    ]

    processed_signals = 0
    # Process file
    with open(csv_path, 'r', newline='') as infile:
        csv_reader = csv.reader(infile)
        for num, row in enumerate(csv_reader):
            try:
                star_idx = row.index('*')
            except ValueError:
                continue

            raw_signal = np.array([float(x) for x in row[:star_idx]], dtype=np.float32)
            info_arr = row[star_idx+1:]
            ID = info_arr[0] if info_arr else ''

            signal = normalize_signal(raw_signal)

            if padding:
                signal, pad_left, pad_right = pad_signal(signal, window_length)
            else:
                pad_left = pad_right = 0

            windows, centers = prepare_windows(signal, window_length, step)

            results = []
            model_device = device
            for i in range(0, len(windows), batch_size):
                batch_windows = windows[i:i+batch_size]
                current_centers = centers[i:i+batch_size]
                batch_array = np.stack(batch_windows, axis=0)
                batch_tensor = torch.tensor(batch_array, dtype=torch.float32, device=model_device)
                with torch.no_grad():
                    outputs = model(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1).cpu().numpy().tolist()
                    probs_cpu = probs.cpu()

                for b_idx in range(len(preds)):
                    klas = int(preds[b_idx])
                    prob = probs_cpu[b_idx:b_idx+1, :]
                    center = current_centers[b_idx]
                    results.append((center, klas, prob))

            klasifikace = [r[1] for r in results]

            # map predicted label indices to gene names, exclude 'no gene' (0) unless no other found
            unique_labels = sorted(set(klasifikace))
            found_labels = [l for l in unique_labels if l != 0]
            if not found_labels:
                found_genes_str = "None"
            else:
                found_genes_str = ";".join(gene_names[l] for l in found_labels)

            results_df.loc[len(results_df)] = {
                'num': f"{num:08d}",
                'ID': ID,
                'found_genes': found_genes_str
            }

            # depad for plotting
            if padding:
                signal_plot_arr = depad_signal(signal, pad_left, pad_right)
                centers_plot = [c - pad_left for c in centers]
            else:
                signal_plot_arr = signal
                centers_plot = centers

            if export_images:
                export_image(signal_plot_arr, centers_plot, results, images_dir, num, ID,
                             None if found_genes_str == "None" else found_genes_str,
                             gene_names, label_colors)

            # update progress bar only in parts (e.g. ~20 updates over the run)
            processed_signals += 1
            if total_signals > 0:
                interval = 10  # every N-th signal; adjust this value as needed
                if processed_signals % interval == 0 or processed_signals == total_signals:
                    percent = processed_signals * 100.0 / total_signals if total_signals else 100.0
                    sys.stdout.write(f"\rProcessing: {processed_signals}/{total_signals} ({percent:.1f}%)")
                    sys.stdout.flush()
            else:
                sys.stdout.write("\rProcessing: 100% \n")
                sys.stdout.flush()

    # finish progress line
    if total_signals > 0:
        print()

    results_csv = os.path.join(out_dir, f"{csv_name}.csv")
    results_df.to_csv(results_csv, index=False)
    return results_csv

if __name__ == "__main__":
    res = NanoResFormer(
        "C:\\Data\\Jakubicek\\NanoGeneNetV2\\Data\\transformer_nanopore_data\\mixed_test_10G_fixed_multilabel_with_ID.csv",
        "C:\\Data\\Jakubicek\\NanoGeneNetV2\\Results_Test",
        csv_name="results_test",
        OV=90,
        Model="middle",
        export_images=True,
        device_pref='auto'
    )
    print("Results saved to:", res)
