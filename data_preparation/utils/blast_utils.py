"""
===============================================================================
Soubor: utils/blast_utils.py
Popis:
    Funkce pro BLAST nástroje:

    • make_blast_db   – vytvoří lokální nukleotidovou DB z FASTA.
    • perform_local_blast – provede BLASTN dotaz a vrátí výsledky jako DataFrame.
Části kódu převzaty a upraveny dle potřeby z nástroje NANOBLAST (viz bakalářská práce).
===============================================================================
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from Bio.Application import ApplicationError
from Bio.Blast.Applications import (
    NcbiblastnCommandline,
    NcbimakeblastdbCommandline,
)


def make_blast_db(
    input_fasta: str | Path,
    output_prefix: str | Path
) -> None:
    """
    Vytvoří BLAST databázi z FASTA.

    Parametry
    ---------
    input_fasta : str | Path
        Zdrojový FASTA soubor.
    output_prefix : str | Path
        Prefix výstupních souborů (.nhr, .nin, ...).
    """
    cmd = NcbimakeblastdbCommandline(
        input_file=str(input_fasta),
        dbtype="nucl",
        out=str(output_prefix),
    )
    try:
        cmd()
    except ApplicationError as exc:
        raise RuntimeError(f"makeblastdb selhalo: {exc.stderr}") from exc


def perform_local_blast(
    query_seq: str,
    database: str | Path,
    output_file: str | Path,
    max_hits: int = 1_000,
    evalue: float = 1e-3
) -> pd.DataFrame:
    """
    Spustí BLASTN nad lokální DB a načte výsledky do pandas.DataFrame.

    Parameters
    ----------
    query_seq : str
        Sekvence dotazu.
    database : str | Path
        Prefix databáze vytvořené pomocí make_blast_db.
    output_file : str | Path
        Kam uložit BLAST tabulku (outfmt 6).
    max_hits : int
        Maximální počet záznamů.
    evalue : float
        E-value threshold.

    Returns
    -------
    pandas.DataFrame
        Výsledná tabulka s 12 sloupci (qseqid, sseqid, pident, … bitscore).
    """
    output_file = Path(output_file)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as qf:
        qf.write(">query\n")
        qf.write(query_seq)

    cmd = NcbiblastnCommandline(
        query=qf.name,
        db=str(database),
        out=str(output_file),
        outfmt=6,
        max_target_seqs=max_hits,
        evalue=evalue,
    )
    try:
        cmd()
    except ApplicationError as exc:
        raise RuntimeError(f"blastn selhalo: {exc.stderr}") from exc

    cols = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    ]
    return pd.read_csv(output_file, sep="\t", header=None, names=cols)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Příklad:
    #
    # make_blast_db("gene.fa", "work/mydb")
    # tbl = perform_local_blast(
    #     query_seq="ATGC...",
    #     database="work/mydb",
    #     output_file="work/blast_out.tsv"
    # )
    # print(tbl.head())
    #
    pass
