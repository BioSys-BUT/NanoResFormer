"""
===============================================================================
Soubor: processing/process_signal.py
Popis:
    Provádí down-sampling signálů a ukládá hotový CSV řádek:
        [<downsampled_signal...>, start_idx, stop_idx, gene_idx]
===============================================================================
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def process_signal_v3(
    input_csv: str | Path,
    output_csv: str | Path,
    gene_idx: int,
    downsampling_factor: int = 8
) -> None:
    """
    Načte CSV se signály, provede down-sampling a uloží nový CSV.

    Parametry
    ---------
    input_csv : str | Path
        CSV z `create_csv_file` (sloupce signal, start_signal…).
    output_csv : str | Path
        Cílový CSV se zpracovanými daty.
    gene_idx : int
        Číselná identifikace genu (použije se jako label).
    downsampling_factor : int
        Každý *n*-tý vzorek zůstane zachován.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists() or input_csv.stat().st_size == 0:
        print(f"Přeskakuji prázdný soubor {input_csv}")
        return

    df = pd.read_csv(input_csv)
    if df.shape[0] < 2:
        print(f"{input_csv}: nedostatek řádků.")
        return

    out_rows: List[List[float | int]] = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            signal = np.array([float(x) for x in str(row["signal"]).split(";") if x])
        except Exception:
            skipped += 1
            continue

        if signal.size == 0:
            skipped += 1
            continue

        start_signal = int(row["start_signal"]) // downsampling_factor
        stop_signal = int(row["stop_signal"]) // downsampling_factor
        down_sig = signal[::downsampling_factor]

        out_rows.append([*down_sig, start_signal, stop_signal, gene_idx])

    if out_rows:
        with Path(output_csv).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(out_rows)
    print(f"{output_csv}: {len(out_rows)} řádků, přeskočeno {skipped}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Příklad:
    #
    # process_signal_v3(
    #     input_csv="work/raw_signals.csv",
    #     output_csv="work/processed_signals.csv",
    #     gene_idx=1,
    #     downsampling_factor=8,
    # )
    #
    pass
