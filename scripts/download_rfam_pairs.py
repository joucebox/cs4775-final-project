#!/usr/bin/env python3
"""Download Rfam alignment families and generate pairwise alignment files."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import requests

from .constants import (
    ALIGNMENTS_FOLDER,
    DEFAULT_NUM_FAMILIES,
    FASTA_FOLDER,
    FULL_ALIGN_BASE_URL,
    MAX_PAIRS_PER_FAMILY,
    MAX_SEQUENCES_PER_FAMILY,
    RANDOM_SEED,
    RFAM_FA_PATH,
)

# Ensure repository modules are importable when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.types import RNASequence
from src.utils import read_rna_fasta
from src.utils.stockholm_writer import (
    parse_stockholm_to_dict,
    write_fasta_pairwise,
    write_stockholm_pairwise,
)


def load_rfam_sequences(fasta_path: Path, ids: List[str]) -> Dict[str, RNASequence]:
    """Load Rfam sequences from FASTA file into a dictionary."""
    sequences = read_rna_fasta(str(fasta_path), ids)
    rfam_dict = {seq.identifier: seq for seq in sequences}
    print(f"[load_rfam] Loaded {len(rfam_dict)} sequences")
    return rfam_dict


def fetch_rfam_alignment_list() -> list[str]:
    """Fetch list of Stockholm alignment files from Rfam FTP."""
    print(f"[fetch_list] Fetching alignment list from {FULL_ALIGN_BASE_URL}")
    try:
        response = requests.get(FULL_ALIGN_BASE_URL, timeout=30)
        response.raise_for_status()

        # Extract RF*.sto filenames using regex
        files = re.findall(r'href="(RF\d+\.sto)"', response.text)
        files = sorted(set(files))

        print(f"[fetch_list] Found {len(files)} alignment files")
        return files
    except Exception as e:
        print(f"[error] Failed to fetch alignment list: {e}")
        raise


def download_stockholm(family_id: str) -> str:
    """Download Stockholm alignment file for a family."""
    url = f"{FULL_ALIGN_BASE_URL}{family_id}.sto"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[error] Failed to download {family_id}: {e}")
        raise


def generate_pairwise_files(
    family_id: str,
    parsed_seqs: Dict[str, str],
    pair_idx: int,
    seq1_name: str,
    seq2_name: str,
    rfam_dict: Dict[str, RNASequence],
) -> None:
    """Generate pairwise .sto and .fa files for a sequence pair."""
    if seq1_name not in parsed_seqs or seq2_name not in parsed_seqs:
        print(f"[warn] Sequences not found in data: {seq1_name}, {seq2_name}")
        return

    aligned1 = parsed_seqs[seq1_name]
    aligned2 = parsed_seqs[seq2_name]

    unaligned1 = "".join(rfam_dict[seq1_name].residues)
    unaligned2 = "".join(rfam_dict[seq2_name].residues)

    # Generate output filenames
    base_name = f"{family_id}_{pair_idx}"
    sto_path = ALIGNMENTS_FOLDER / f"{base_name}.sto"
    fa_path = FASTA_FOLDER / f"{base_name}.fa"

    # Write Stockholm file
    write_stockholm_pairwise(
        str(sto_path),
        seq1_name,
        seq2_name,
        aligned1,
        aligned2,
    )

    # Write FASTA file
    write_fasta_pairwise(
        str(fa_path),
        seq1_name,
        seq2_name,
        unaligned1,
        unaligned2,
    )


def main() -> None:
    """Main function to download and process Rfam alignments."""
    # Set random seed
    random.seed(RANDOM_SEED)

    # Create output directories
    ALIGNMENTS_FOLDER.mkdir(parents=True, exist_ok=True)
    FASTA_FOLDER.mkdir(parents=True, exist_ok=True)

    # Fetch list of alignment files
    try:
        all_families = fetch_rfam_alignment_list()
    except Exception:
        print("[error] Failed to fetch alignment list. Exiting.")
        sys.exit(1)

    if not all_families:
        print("[error] No alignment files found. Exiting.")
        sys.exit(1)

    # The target is exactly DEFAULT_NUM_FAMILIES unique, valid families
    print(
        f"[target] Fetching {DEFAULT_NUM_FAMILIES} families out of {len(all_families)} available"
    )

    # Shuffle all families to form a random order
    families_shuffled = all_families[:]  # copy to avoid modifying the original
    random.shuffle(families_shuffled)
    family_ids = [f.replace(".sto", "") for f in families_shuffled]

    print(f"[config] Max pairs per family: {MAX_PAIRS_PER_FAMILY}")
    print(f"[config] Random seed: {RANDOM_SEED}\n")

    # Step 1: Download Stockholm files and collect sequence IDs needed
    print("[step 1] Downloading Stockholm files and collecting sequence IDs...")
    family_data = {}  # family_id -> {seq_id: aligned_seq}
    all_seq_ids = set()

    # Iterate through the shuffled families, stopping when DEFAULT_NUM_FAMILIES are collected
    for idx, family_id in enumerate(family_ids, 1):
        print(f"  [{idx}/{len(families_shuffled)}] Downloading {family_id}...")
        try:
            sto_content = download_stockholm(family_id)
            parsed_seqs = parse_stockholm_to_dict(sto_content)

            if len(parsed_seqs) >= 2 and len(parsed_seqs) <= MAX_SEQUENCES_PER_FAMILY:
                family_data[family_id] = parsed_seqs
                all_seq_ids.update(parsed_seqs.keys())
                print(f"    Found {len(parsed_seqs)} sequences")
            else:
                print(
                    f"    [skip] Found {len(parsed_seqs)} sequence(s), skipping family"
                )
            if len(family_data) >= DEFAULT_NUM_FAMILIES:
                break
        except Exception as e:
            print(f"    [error] Failed to download {family_id}: {e}")
            continue

    if len(family_data) < DEFAULT_NUM_FAMILIES:
        print(
            f"[warning] Only {len(family_data)} valid families downloaded; "
            f"{DEFAULT_NUM_FAMILIES} requested. Insufficient valid families found."
        )

    print(
        f"\n[step 1 complete] Found {len(family_data)} families.",
        f"Total unique sequence IDs needed: {len(all_seq_ids)}",
    )

    # Step 2: Load only the needed sequences from Rfam.fa
    print(f"\n[step 2] Loading {len(all_seq_ids)} sequences from  fasta file...")
    if not RFAM_FA_PATH.exists():
        print(f"[error] Rfam FASTA file not found: {RFAM_FA_PATH}")
        sys.exit(1)

    rfam_dict = load_rfam_sequences(RFAM_FA_PATH, list(all_seq_ids))
    print(f"[step 2 complete] Loaded {len(rfam_dict)} sequences")

    # Step 3: Generate pairwise alignments
    print(f"\n[step 3] Generating pairwise alignments...")
    total_pairs = 0

    for idx, (family_id, parsed_seqs) in enumerate(family_data.items(), 1):
        print(f"  [{idx}/{len(family_data)}] Processing {family_id}...")

        # Filter to only sequences we have in rfam_dict
        seq_names = [k for k in parsed_seqs.keys() if k in rfam_dict]

        if len(seq_names) < 2:
            print(f"    [skip] Only {len(seq_names)} sequence(s) with FASTA data")
            continue

        # Generate all possible pairs
        all_pairs = [
            (i, j) for i in range(len(seq_names)) for j in range(i + 1, len(seq_names))
        ]

        # Sample pairs if necessary
        if MAX_PAIRS_PER_FAMILY and len(all_pairs) > MAX_PAIRS_PER_FAMILY:
            sampled_pairs = random.sample(all_pairs, MAX_PAIRS_PER_FAMILY)
        else:
            sampled_pairs = all_pairs

        print(f"    {len(seq_names)} seqs -> {len(sampled_pairs)} pairs")

        for pair_idx, (i, j) in enumerate(sampled_pairs):
            seq1_name = seq_names[i]
            seq2_name = seq_names[j]

            try:
                generate_pairwise_files(
                    family_id,
                    parsed_seqs,
                    pair_idx,
                    seq1_name,
                    seq2_name,
                    rfam_dict,
                )
            except Exception as e:
                print(f"    [error] Failed to generate pair {pair_idx}: {e}")
                continue

        total_pairs += len(sampled_pairs)

    print(
        f"\n[done] Generated {total_pairs} pairwise alignments from {len(family_data)} families"
    )
    print("[output] Files written to:")
    print(f"  - {ALIGNMENTS_FOLDER}")
    print(f"  - {FASTA_FOLDER}")


if __name__ == "__main__":
    main()
