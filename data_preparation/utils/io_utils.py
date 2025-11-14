"""
===============================================================================
Soubor: utils/io_utils.py
Popis:
    Sada pomocných funkcí pro převod mezi formáty BAM / SAM / FASTA.

Funkce:
    merge_bam_files_in_directory_to_sam
        Sloučí všechny *.bam v zadaném adresáři do jednoho SAM.
    sam_to_fasta_convert
        Vytvoří FASTA se sekvencemi ze SAM.

Vyžaduje nástroj `samtools` dostupný v PATH.
Části kódu převzaty a upraveny dle potřeby z nástroje NANOBLAST (viz bakalářská práce).
===============================================================================
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def merge_bam_files_in_directory_to_sam(
    directory: str | Path,
    output_sam: str | Path
) -> None:
    """
    Sloučí všechny BAM soubory v adresáři do jednoho SAM souboru.

    Parameters
    ----------
    directory : str | Path
        Adresář obsahující BAM soubory (např. .../bam_pass/barcode31)
    output_sam : str | Path
        Výstupní SAM.

    Raises
    ------
    FileNotFoundError
        Pokud v adresáři nejsou žádné *.bam.
    """
    directory = Path(directory)
    output_sam = Path(output_sam)

    bam_files = sorted(p for p in directory.iterdir() if p.suffix == ".bam")
    if not bam_files:
        raise FileNotFoundError(f"V {directory} nebyly nalezeny žádné BAM soubory.")

    # zapíšeme hlavičku z prvního BAM
    with output_sam.open("w") as sam_fh:
        subprocess.run(
            ["samtools", "view", "-H", str(bam_files[0])],
            stdout=sam_fh,
            check=True,
        )

    # a postupně připojíme záznamy všech BAM
    with output_sam.open("a") as sam_fh:
        for bam in bam_files:
            subprocess.run(
                ["samtools", "view", str(bam)],
                stdout=sam_fh,
                check=True,
            )

    print(f"Sloučeno {len(bam_files)} BAM souborů do {output_sam}")


def sam_to_fasta_convert(
    input_sam: str | Path,
    output_fasta: str | Path
) -> None:
    """
    Převede SAM na FASTA (read-id → sekvence).

    Parameters
    ----------
    input_sam : str | Path
        Vstupní SAM.
    output_fasta : str | Path
        Cílový FASTA soubor.
    """
    input_sam = Path(input_sam)
    output_fasta = Path(output_fasta)

    with input_sam.open() as sam_fh, output_fasta.open("w") as fasta_fh:
        for line in sam_fh:
            if line.startswith("@"):
                continue
            parts = line.rstrip("\n").split("\t")
            read_id = parts[0]   # QNAME
            seq = parts[9]       # SEQ
            fasta_fh.write(f">{read_id}\n{seq}\n")

    print(f"FASTA uloženo do {output_fasta}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Příklady použití:
    #
    # merge_bam_files_in_directory_to_sam(
    #     directory="test_data/bam_pass/barcode31",
    #     output_sam="work/merged_bc31.sam",
    # )
    #
    # sam_to_fasta_convert(
    #     input_sam="work/merged_bc31.sam",
    #     output_fasta="work/merged_bc31.fasta",
    # )
    #
    pass
