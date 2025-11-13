# NanoResFormer

A hybrid convolutional-transformer neural network for real-time antimicrobial resistance detection directly from Oxford Nanopore Technologies (ONT) raw signal data.

![NanoResFormer](Image.png "NanoResFormer workflow")

## Overview

NanoResFormer enables basecalling-free diagnostics by analyzing raw nanopore current signals (squiggles) to detect antimicrobial resistance genes during sequencing. This approach significantly reduces time-to-diagnosis and computational requirements compared to conventional basecalling-dependent workflows.

## Key Features

- **Direct Signal Analysis**: Processes raw ONT current signals without basecalling
- **High Accuracy**: Achieves 94.2% sensitivity and 3.2% false positive rate for resistance gene detection
- **Fast Processing**: Screens 1 million reads in 6.3 hours (~50% faster than basecalling alone)
- **Rapid Diagnosis**: Preliminary results available within 9 minutes of sequencing (24,000 reads)
- **Hybrid Architecture**: Combines convolutional layers with transformer-based attention mechanisms

## Clinical Applications

- Real-time antimicrobial resistance screening during sequencing
- Point-of-care diagnostics in clinical microbiology
- Rapid on-site clinical decision support
- Resource-efficient pathogen surveillance

## Installation

Due to flexibility across different devices (GPU/CPU), PyTorch is not included in `requirements.txt`.
**You must install it manually BEFORE running the application.**

1. **Install other dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

2. **Install PyTorch:**
  Visit the official website https://pytorch.org/ and choose the version matching your **CUDA version** and operating system.

  * **For CPU version (universal):**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
  * **For NVIDIA GPU (example for CUDA 12.1):**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

## Usage

```python
from nanoresformer import NanoResFormer

# Initialize model
model = NanoResFormer(
   input_dim=512,
   hidden_dim=256,
   num_layers=6
)

# Process raw signal data
predictions = model(raw_signal_data)
```

## Performance

- **Sensitivity**: 94.2%
- **False Positive Rate**: 3.2%
- **Processing Speed**: 1M reads in 6.3 hours
- **Time to First Detection**: ~9 minutes (99.9% confidence)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- ONT raw signal data (.fast5 or .pod5 format)
- Additional dependencies in `requirements.txt`

## Future Development

- Continuous real-time processing with adaptive data stream handling
- Expanded resistance gene database coverage
- Optimized transformer variants for reduced computational demands
- Integration into ONT sequencing devices for autonomous diagnostics

## License

See LICENSE file for details.

## Citation

If you use NanoResFormer in your research, please cite:

```bibtex
@software{nanoresformer,
  title={Basecalling-free resistance gene identification using a hybrid transformer in raw nanopore signals},
  author={Jakubicek R. et al.},
  year={2026}
}
```

## Contact

For questions and feedback, please open an issue on GitHub or contact us via email at [jakubicek@vutbr.cz](mailto:jakubicek@vutbr.cz) or [jakubickova@vut.cz](mailto:jakubickova@vut.cz).