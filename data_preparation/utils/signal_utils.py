"""
===============================================================================
Soubor: utils/signal_utils.py
Popis:
    Funkce pro vyhledávání úseků signálu a načítání z POD5/FAST5.

Klíčové funkce:
    extract_info_from_tsv
    sam_file_signal_search
    load_signals
    dump_signals_csv / dump_signals_parquet
Části kódu převzaty a upraveny dle potřeby z nástroje NANOBLAST (viz bakalářská práce).
===============================================================================
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Set, Tuple, Union

import h5py
import pod5 as p5
import pyarrow as pa
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# 1. Načtení informací z TSV (BLAST výstup)
# ─────────────────────────────────────────────────────────────────────────────

def extract_info_from_tsv(
    filename: str | Path
) -> Dict[str, Tuple[int, int, int, float]]:
    """
    Načte TSV (outfmt 6) a vrátí slovník:
        {read_id: (sstart, send, délka_genu, bitscore)}.
    """
    with Path(filename).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {
            row["sseqid"]: (
                int(row["sstart"]),
                int(row["send"]),
                int(row["length"]),
                float(row["bitscore"]),
            )
            for row in reader
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vyhledání signálů v SAM souboru
# ─────────────────────────────────────────────────────────────────────────────

def _signal_searcher(
    move_list: Iterable[int],
    start_idx: int,
    stop_idx: int,
    trimmed_samples: int,
    stride: int = 5,
) -> Tuple[int | None, int | None]:
    """
    Interní: podle move_table nalezne indexy signálu v pA prostoru.
    """
    count_all = count_ones = 0
    start_signal = stop_signal = None

    for mv in move_list:
        count_all += 1
        if mv != 1:
            continue
        count_ones += 1
        if count_ones == start_idx:
            start_signal = count_all * stride + trimmed_samples
        if count_ones == stop_idx:
            stop_signal = count_all * stride + trimmed_samples
            break
    return start_signal, stop_signal


def sam_file_signal_search(
    sam_filename: str | Path,
    read_ids: Union[Set[str], str],
    signal_ranges: Dict[str, Tuple[int, int, int, float]],
    mode: Literal["parquet", "csv"] = "parquet",
    min_gene_length: float = 0.95,
    min_bitscore: float = 0.0,
) -> Dict[str, Tuple]:
    """
    Z daného SAM souboru vytáhne pro každé read-ID start/stop signálu.

    Parameters
    ----------
    sam_filename : str | Path
        Cesta k SAM souboru.
    read_ids : set[str] | str
        Read-ID, která nás zajímají (obvykle klíče z `extract_info_from_tsv()`).
    signal_ranges : dict
        Informace o genech (sstart, send, len, bitscore).
    mode : {"parquet","csv"}
        Určuje formát navrátileného záznamu (kratší vs. rozšířený).
    min_gene_length : float
        Požadovaný poměr délky signálu ku délce genu.
    min_bitscore : float
        Prahová hodnota Bitscore.

    Returns
    -------
    dict[read_id] -> Tuple | None
        Konkrétní struktura závisí na `mode`.
    """
    def parquet_tpl():  # pouze start, stop
        return start_signal, stop_signal

    def csv_tpl():      # rozšířená verze pro CSV export
        return start_signal, stop_signal, sequence, start_idx, stop_idx, direction

    result_func = {"parquet": parquet_tpl, "csv": csv_tpl}[mode]

    if isinstance(read_ids, str):
        read_ids = {read_ids}

    result: dict[str, Tuple | None] = {rid: None for rid in read_ids}

    def peek_line(fh):
        pos = fh.tell()
        line = fh.readline()
        fh.seek(pos)
        return line

    def move_table_parser(mtable: str) -> Iterable[int]:
        for ch in mtable:
            if ch.isdigit():
                yield int(ch)

    with Path(sam_filename).open() as sam:
        while peek_line(sam).startswith("@"):
            sam.readline()

        for line in sam:
            parts = line.rstrip("\n").split("\t")
            qname = parts[0]
            if qname not in read_ids:
                continue

            start_idx, stop_idx, gene_len, bitscore = signal_ranges[qname]
            sequence = parts[9]

            direction = 1
            move_table = None
            stride = 5
            trimmed_samples = 0

            for field in parts[11:]:
                if field.startswith("mv:B:c"):
                    move_table = field[9:]
                    stride = int(field[7])
                elif field.startswith("ts:i:"):
                    trimmed_samples = int(field[5:])

            if start_idx > stop_idx:
                start_idx, stop_idx = stop_idx, start_idx
                direction = -1

            if not move_table:
                continue

            start_signal, stop_signal = _signal_searcher(
                move_table_parser(move_table),
                start_idx, stop_idx,
                trimmed_samples, stride,
            )
            if start_signal is None or stop_signal is None:
                continue

            if (stop_signal - start_signal) < min_gene_length * gene_len:
                continue
            if bitscore < min_bitscore:
                continue

            result[qname] = result_func()
            if all(v is not None for v in result.values()):
                break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Načítání signálů z POD5 / FAST5
# ─────────────────────────────────────────────────────────────────────────────

def _load_from_pod5(
    file_path: Path,
    signal_ranges,
    mode: Literal["parquet", "csv"]
):
    def parquet_tpl():
        return signal_in_pa[start:stop]

    def csv_tpl():
        return start, stop, *rest, signal_in_pa

    result_func = {"parquet": parquet_tpl, "csv": csv_tpl}[mode]
    result = {}

    with p5.Reader(str(file_path)) as reader:
        for read in reader.reads(signal_ranges.keys(), missing_ok=True):
            signal_in_pa = read.signal_pa
            rid = str(read.read_id)
            if signal_ranges[rid] is None:
                continue
            start, stop, *rest = signal_ranges[rid]
            result[rid] = result_func()
    return result


def _load_from_fast5(
    file_path: Path,
    signal_ranges,
    mode: Literal["parquet", "csv"]
):
    def parquet_tpl():
        return signal_in_pa[start:stop]

    def csv_tpl():
        return start, stop, *rest, signal_in_pa

    result_func = {"parquet": parquet_tpl, "csv": csv_tpl}[mode]
    result = {}

    with h5py.File(file_path, "r") as fh:
        for read in fh:
            rid = fh[f"{read}/Raw"].attrs["read_id"].decode()
            if rid not in signal_ranges or signal_ranges[rid] is None:
                continue

            raw_signal = fh[f"{read}/Raw/Signal"][:]
            ch = fh[f"{read}/channel_id"]
            signal_in_pa = (raw_signal + ch.attrs["offset"]) * ch.attrs["range"] / ch.attrs["digitisation"]

            start, stop, *rest = signal_ranges[rid]
            result[rid] = result_func()
    return result


def load_signals(
    path_list: Union[str, List[str]],
    signal_ranges,
    mode: Literal["parquet", "csv"] = "parquet"
):
    """
    Rekurzivně projde zadané soubory/adresáře a načte signály.

    Parameters
    ----------
    path_list : str | List[str]
        Jeden nebo více souborů / složek k prohledání.
    signal_ranges : dict
        Úseky signálu z `sam_file_signal_search`.
    mode : {"parquet","csv"}
        Určuje strukturu vracených hodnot.

    Returns
    -------
    dict[read_id] -> Tuple | None
    """
    if isinstance(path_list, str):
        path_list = [path_list]

    result = {rid: None for rid in signal_ranges.keys()}
    queue = list(map(Path, path_list))

    while queue:
        current = queue.pop()
        if current.is_dir():
            queue.extend(current.iterdir())
            continue

        if current.suffix == ".pod5":
            result.update(_load_from_pod5(current, signal_ranges, mode))
        elif current.suffix == ".fast5":
            result.update(_load_from_fast5(current, signal_ranges, mode))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Export do CSV / Parquet
# ─────────────────────────────────────────────────────────────────────────────

def dump_signals_csv(signals, filename: str | Path) -> None:
    """
    Zapíše signály do CSV; používá se hlavně pro další zpracování
    v process_signal_v3.
    """
    with Path(filename).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["read_id", "sequence", "start_idx", "stop_idx",
             "signal", "start_signal", "stop_signal", "direction"]
        )
        for read_id, result in signals.items():
            if result is None:
                continue
            start_signal, stop_signal, sequence, s_idx, e_idx, direction, signal = result
            writer.writerow(
                [
                    read_id,
                    sequence[s_idx:e_idx + 1],
                    s_idx,
                    e_idx,
                    ";".join(map(str, signal)),
                    start_signal,
                    stop_signal,
                    direction,
                ]
            )


def dump_signals_parquet(signals, _, filename: str | Path) -> None:
    """
    Uloží (read_id, signal) do formátu Parquet – vhodné pro velké objemy dat.
    """
    filtered = {k: v for k, v in signals.items() if v is not None}
    table = pa.table({"read_id": list(filtered.keys()),
                      "signal": list(filtered.values())})
    pq.write_table(table, str(filename))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Krátká ukázka:
    #
    # info = extract_info_from_tsv("result/gene.tsv")
    # rngs = sam_file_signal_search("reads.sam", info.keys(), info, mode="csv")
    # sigs = load_signals("pod5_pass/barcode30", rngs, mode="csv")
    # dump_signals_csv(sigs, "work/signals.csv")
    #
    pass
