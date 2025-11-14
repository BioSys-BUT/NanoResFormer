"""
===============================================================================
Soubor: processing/generate_training_data.py
Popis:
    Hlavní skript – projde zadané barcody a geny, spustí BLAST,
    vyfiltruje čtení podle e-value, délky shody i bitscore
    a vytvoří finální dataset.
Vytvořeno s asistencí chatGPT(model 4-o)
===============================================================================
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
from typing import Iterable

from Bio import SeqIO

from priprava_dat import config
from priprava_dat.utils.io_utils import merge_bam_files_in_directory_to_sam, sam_to_fasta_convert
from priprava_dat.utils.blast_utils import make_blast_db, perform_local_blast
from priprava_dat.processing.create_csv_from_tsv import create_csv_file
from priprava_dat.processing.process_signal import process_signal_v3


def _remove_files(pattern: str) -> None:
    """Pomocná funkce: smaže všechny soubory odpovídající glob-patternu."""
    for fp in glob.glob(pattern):
        try:
            os.remove(fp)
        except OSError:
            pass


def _get_query_sequence(gene_name: str) -> str:
    """
    Najde záznam `gene_name` v jediném FASTA souboru definovaném v configu
    a vrátí jeho sekvenci jako řetězec.
    """
    for rec in SeqIO.parse(config.CHOSEN_GENES_FASTA, "fasta"):
        if rec.id == gene_name:
            return str(rec.seq)
    raise FileNotFoundError(
        f"Záznam {gene_name} nebyl nalezen v {config.CHOSEN_GENES_FASTA}"
    )


def generate_training_data(
    barcodes: Iterable[int],
    gene_names: list[str],
    *,
    downsampling_factor: int = config.DOWNSAMPLING_FACTOR,
    min_gene_length: float = config.MIN_GENE_LENGTH,
    min_bitscore: float = config.MIN_BITSCORE,
    workdir: Path = config.WORKDIR,
) -> None:
    """
    Kompletní pipeline.

    Parametry
    ---------
    barcodes : Iterable[int]
        Seznam / rozsah barcode čísel.
    gene_names : list[str]
        Id záznamů FASTA, které chceme hledat.
    downsampling_factor : int
        Faktor pro down-sampling signálu.
    min_gene_length : float
        Požadovaný poměr délky nalezené shody vůči zadávané sekvenci.
    min_bitscore : float
        Minimální BLAST bitscore, aby byl hit akceptován.
    workdir : Path
        Pracovní adresář pro mezisoubory.
    """
    final_csv = config.FINAL_DATASET
    with final_csv.open("w", newline="") as out_f:
        writer = csv.writer(out_f)

        for bc in barcodes:
            print(f"--- Zpracovávám barcode {bc} ---")
            bam_dir = config.BAM_PASS_DIR / f"barcode{bc}"
            pod5_dir = config.POD5_PASS_DIR / f"barcode{bc}"

            sam_file = workdir / f"merged_bc{bc}.sam"
            fasta_file = sam_file.with_suffix(".fasta")
            db_prefix = workdir / f"blast_db_bc{bc}"

            merge_bam_files_in_directory_to_sam(bam_dir, sam_file)
            sam_to_fasta_convert(sam_file, fasta_file)
            make_blast_db(fasta_file, db_prefix)

            for gene_idx, gene_name in enumerate(gene_names, 1):
                tsv_out = workdir / f"{gene_name}.tsv"
                query_seq = _get_query_sequence(gene_name)

                tbl = perform_local_blast(
                    query_seq, db_prefix, tsv_out,
                    max_hits=config.MAX_BLAST_HITS,
                    evalue=config.BLAST_EVALUE,
                )
                if tbl.empty:
                    print(f"   Bez shody pro {gene_name}")
                    continue

                csv_tmp = workdir / f"bc{bc}_{gene_name}.csv"
                create_csv_file(
                    sam_file, pod5_dir, tsv_out, csv_tmp,
                    min_gene_length=min_gene_length,
                    min_bitscore=min_bitscore,
                )

                processed_csv = csv_tmp.with_stem(csv_tmp.stem + "_processed")
                process_signal_v3(
                    csv_tmp, processed_csv, gene_idx, downsampling_factor
                )

                with processed_csv.open() as in_f:
                    for row in csv.reader(in_f):
                        if row:
                            writer.writerow(row)

            # úklid mezisouborů
            _remove_files(str(workdir / "*.tsv"))
            _remove_files(str(workdir / "*.csv"))
            sam_file.unlink(missing_ok=True)
            fasta_file.unlink(missing_ok=True)

    print(f"Pipeline dokončena – dataset uložen v {final_csv}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Příklad:
    #
    # generate_training_data(
    #     barcodes=range(72, 85),
    #     gene_names=[
    #         "aac(3)-IIa_1_X51534",
    #         "tet(A)_2_X00006",
    #         "aph(6)-Id_1_M28829",
    #     ],
    #     downsampling_factor=1,
    #     min_gene_length=0.95,
    #     min_bitscore=1_373,
    # )
    #
    pass
