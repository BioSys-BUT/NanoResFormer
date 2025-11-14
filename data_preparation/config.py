"""
===============================================================================
Soubor: config.py
Popis:
    Centrální nastavení cest a parametrů pro celou pipeline.
    Ostatní moduly tyto proměnné pouze importují.
    Jsou zde uvedené příklady cest, které byly použity v naší práci.
===============================================================================
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1) Vstupní data z MinIONu
# --------------------------------------------------------------------------- #

EXPERIMENT_DIR: Path = Path(
    "/data/gecko/bioinf_shared/minion_storage/240216-FNB-P2S/"
    "ONT/20240216_1332_P2S-01541-A_PAQ63858_33cbe55b"
)
# Příklad původní cesty viz výše.

BAM_PASS_DIR: Path = EXPERIMENT_DIR / "bam_pass"   # podadresáře barcodeXX
POD5_PASS_DIR: Path = EXPERIMENT_DIR / "pod5_pass"  # podadresáře barcodeXX

# --------------------------------------------------------------------------- #
# 2) FASTA soubory sledovaných genů
# --------------------------------------------------------------------------- #

CHOSEN_GENES_FASTA: Path = Path(
    "/home/vorochta/verze_testu_pro_server/chosen_genes.fasta"
)
# cesta k fasta souboru, který obsahuje záznamy genů, ze kterých můžeme vybírat ty, které chceme hledat

# --------------------------------------------------------------------------- #
# 3) Pracovní adresář a výstupy
# --------------------------------------------------------------------------- #

WORKDIR: Path = Path("/home/user_pool_2/vorochta/pipeline_work")
WORKDIR.mkdir(exist_ok=True, parents=True)

FINAL_DATASET: Path = WORKDIR / "final_training_data.csv"

# --------------------------------------------------------------------------- #
# 4) Další parametry
# --------------------------------------------------------------------------- #
DOWNSAMPLING_FACTOR: int = 8 #faktor podvzorkování signálu (pokud nechcete podvzorkovávat, uveďte int = 1)
MAX_BLAST_HITS: int = 1_000 #maximální počet BLAST hitů
BLAST_EVALUE: float = 1e-3 #minimální e-value pro filtraci BLAST výsledků
MIN_GENE_LENGTH: float = 0.95   #minimální délka BLAST hitu vůči zadávané sekvenci
MIN_BITSCORE: float = 1_373      # minimální bitscore BLAST hitu
