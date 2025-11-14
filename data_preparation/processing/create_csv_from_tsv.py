"""
===============================================================================
Soubor: processing/create_csv_from_tsv.py
Popis:
    Z kombinace SAM + POD5/FAST5 + TSV (výběr genu) vytvoří CSV se signály.
    Spouští se v samostatném vlákně, aby neblokovalo hlavní pipeline.
===============================================================================
"""
from __future__ import annotations

from pathlib import Path
import threading

import priprava_dat.config
from priprava_dat.utils.signal_utils import (
    extract_info_from_tsv,
    sam_file_signal_search,
    load_signals,
    dump_signals_csv,
)


def create_csv_file(
    sam_file: str | Path,
    signals_dir: str | Path,
    tsv_file: str | Path,
    output_csv: str | Path,
    *,
    min_gene_length: float = config.MIN_GENE_LENGTH,
    min_bitscore: float = config.MIN_BITSCORE
) -> None:
    """
    Kombinuje SAM + POD5/FAST5 + TSV → CSV se signálem, s filtry
    `min_gene_length` a `min_bitscore`.

    Parametry
    ---------
    sam_file, signals_dir, tsv_file, output_csv : Path
        Cesty k jednotlivým souborům.
    min_gene_length : float
        Požadovaný poměr délky nalezené shody vůči zadávané sekvenci.
    min_bitscore : float
        Minimální bit-skóre z BLASTu.
    """
    if not all(Path(p).exists() for p in (sam_file, signals_dir, tsv_file)):
        raise FileNotFoundError("Chybí vstupní soubor nebo složka.")

    def worker():
        info = extract_info_from_tsv(tsv_file)
        ranges = sam_file_signal_search(
            sam_file, info.keys(), info, mode="csv",
            min_gene_length=min_gene_length,
            min_bitscore=min_bitscore,
        )
        signals = load_signals(signals_dir, ranges, mode="csv")
        dump_signals_csv(signals, output_csv)
        print(f"CSV vytvořeno → {output_csv}")

    threading.Thread(target=worker, daemon=True).start()

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Příklad:
    #
    # create_csv_file(
    #     sam_file="work/merged_bc30.sam",
    #     signals_dir="/path/to/pod5_pass/barcode30",
    #     tsv_file="work/tetA.tsv",
    #     output_csv="work/tetA_signals.csv",
    # )
    #
    pass
