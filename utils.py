import os
import numpy as np
import torch
import psutil
import matplotlib.pyplot as plt


def normalize_signal(signal):
    signal = np.array(signal, dtype=np.float32)
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    if std_val != 0:
        return (signal - mean_val) / std_val
    return signal - mean_val

def determine_batch_size(window_length, device_pref='auto'):
    # decide batch size based on GPU memory (preferred) or available RAM
    try:
        batch_size = 1
        if (device_pref in ('auto', 'cuda')) and torch.cuda.is_available():
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            total_bytes = props.total_memory
            overhead_factor = 6
            bytes_per_sample = window_length * 4 * overhead_factor
            batch_size = max(1, int((total_bytes * 0.5) / bytes_per_sample))
            batch_size = min(batch_size, 256)
        else:
            try:
                avail = psutil.virtual_memory().available
                overhead_factor = 2
                bytes_per_sample = window_length * 4 * overhead_factor
                batch_size = max(1, int((avail * 0.3) / bytes_per_sample))
                batch_size = min(batch_size, 64)
            except Exception:
                batch_size = 1
    except Exception:
        batch_size = 1
    return int(batch_size)

def pad_signal(signal, window_length):
    pad_left = window_length // 2
    pad_right = window_length - pad_left - 1
    padded = np.pad(signal, (pad_left, pad_right), mode='constant')
    return padded, pad_left, pad_right

def depad_signal(padded_signal, pad_left, pad_right):
    if pad_right > 0:
        return padded_signal[pad_left:-pad_right]
    return padded_signal[pad_left:]

def model_mapping(Model):
    Model = Model.lower()
    if Model == "low":
        from models.transformer_model_LOW import FullModel
        model_name = "model_LOW"
        d_model = 8
    elif Model == "middle":
        from models.transformer_model_MIDDLE import FullModel
        model_name = "model_MIDDLE"
        d_model = 64
    elif Model == "high":
        from models.transformer_model_HIGH import FullModel
        model_name = "model_HIGH"
        d_model = 64
    else:
        raise ValueError("Model must be 'low', 'middle' or 'high'.")
    return FullModel, model_name, d_model

def select_device(device_pref='auto'):
    if device_pref == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_pref == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != 'cuda':
            print("Warning: CUDA requested but not available, falling back to CPU.")
        return device
    elif device_pref == 'cpu':
        return torch.device("cpu")
    else:
        raise ValueError("device_pref must be 'auto', 'cpu' or 'cuda'")

def prepare_windows(signal, window_length, step):
    positions = []
    pos = 0
    L = len(signal)
    while pos + window_length <= L:
        positions.append(pos)
        pos += step
    if pos < L:
        positions.append(max(0, L - window_length))
    windows = []
    centers = []
    for pos in positions:
        window = signal[pos:pos + window_length]
        windows.append(window.astype(np.float32))
        centers.append(pos + window_length // 2)
    return windows, centers

def export_image(signal_plot_arr, centers_plot, results, images_dir, num, ID, found_genes, gene_names, label_colors):
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(range(len(signal_plot_arr)), signal_plot_arr, label='Signal', alpha=0.5)
    ax1.set_xlabel('Position in signal')
    ax1.set_ylabel('Standardized signal (z-score)')
    ax1.set_ylim(-3.0, 3.5)
    ax1.set_xlim(0, len(signal_plot_arr))

    ax2 = ax1.twinx()
    num_classes = len(gene_names)
    for class_idx in range(1, num_classes):
        prob_values = [r[2][0, class_idx].item() for r in results]
        ax2.plot(centers_plot, prob_values, label=gene_names[class_idx], alpha=0.7, color=label_colors[class_idx])
    prob_no_gene = [r[2][0, 0].item() for r in results]
    ax2.plot(centers_plot, prob_no_gene, label='no gene', color='gray', linestyle='--', alpha=0.8)

    ax2.set_ylabel('Detection Probability')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(0, len(signal_plot_arr))
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.0, alpha=1.0)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    title_found = found_genes if found_genes is not None else "None"
    plt.title(f"Signal {num:08d} ID={ID} Predicted={title_found}")
    out_png = os.path.join(images_dir, f"signal_{num:08d}.png")
    plt.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
