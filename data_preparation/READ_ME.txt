How to Run the Pipeline

Edit the paths in the config.py file to match your system.
(You only need to change one file – other code loads from it.)
Install required libraries into your environment:
pip install biopython pandas numpy pyarrow pod5 h5py

Make sure you have the following external tools available in your system:
• samtools
• NCBI BLAST+ (makeblastdb, blastn)


Open the main script in the processing/generate_training_data directory,
choose all parameters in the main section and run the script:
Example:
generate_training_data(
barcodes=range(72, 85),
gene_names=[
"aac(3)-IIa_1_X51534",
"tet(A)_2_X00006",
"aph(6)-Id_1_M28829",
],
downsampling_factor=1,
min_gene_length=0.95,
min_bitscore=1_373,
)
The output will be a final_training_data.csv file in the directory set
in config.WORKDIR.


Used Libraries (Python 3.12)
biopython
pandas
numpy
pyarrow
pod5
h5py
(Python standard libraries – csv, pathlib, subprocess, tempfile, glob,
argparse, threading, os, typing)