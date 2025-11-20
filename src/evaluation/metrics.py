"""Alignment evaluation metrics and helpers."""

from __future__ import annotations

from itertools import zip_longest
from typing import Callable, List, NamedTuple, Set, Tuple

from src.types import Alignment, SequenceType

PAIRWISE_COUNT = 2
GAP_CHARACTERS = {"-", "."}
MetricFunction = Callable[[Alignment, Alignment], float]


class _PairConfusion(NamedTuple):
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


def _ensure_pairwise_alignment(
    alignment: Alignment,
) -> Tuple[SequenceType, SequenceType]:
    """Validate that the alignment contains exactly two sequences."""
    if alignment.num_sequences != PAIRWISE_COUNT:
        raise ValueError(
            f"Expected pairwise alignment with {PAIRWISE_COUNT} sequences, "
            f"received {alignment.num_sequences}."
        )
    seq_x, seq_y = alignment.aligned_sequences
    return seq_x, seq_y


def _pair_metadata(
    alignment: Alignment,
) -> Tuple[SequenceType, SequenceType, Tuple[str, str], Tuple[int, int]]:
    """Return aligned sequences alongside their identifiers and lengths."""
    seq_x, seq_y = _ensure_pairwise_alignment(alignment)

    originals = alignment.original_sequences
    if originals and len(originals) == PAIRWISE_COUNT:
        ids = (originals[0].identifier, originals[1].identifier)
        lengths = (len(originals[0]), len(originals[1]))
    else:
        ids = (seq_x.identifier, seq_y.identifier)
        lengths = (
            sum(residue not in GAP_CHARACTERS for residue in seq_x.residues),
            sum(residue not in GAP_CHARACTERS for residue in seq_y.residues),
        )

    return seq_x, seq_y, ids, lengths


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with zero-denominator protection."""
    return numerator / denominator if denominator else 0.0


def _pair_confusion_matrix(predicted: Alignment, gold: Alignment) -> _PairConfusion:
    """Compute pairwise confusion counts between predicted and gold alignments."""
    _, _, pred_ids, pred_lengths = _pair_metadata(predicted)
    _, _, gold_ids, gold_lengths = _pair_metadata(gold)

    if pred_ids != gold_ids:
        raise ValueError(
            "Predicted and gold alignments reference different sequence identifiers: "
            f"{pred_ids} vs {gold_ids}"
        )

    if pred_lengths != gold_lengths:
        raise ValueError(
            "Predicted and gold alignments have inconsistent sequence lengths: "
            f"{pred_lengths} vs {gold_lengths}"
        )

    pred_pairs = extract_aligned_pairs(predicted)
    gold_pairs = extract_aligned_pairs(gold)

    true_positive = len(pred_pairs & gold_pairs)
    false_positive = len(pred_pairs - gold_pairs)
    false_negative = len(gold_pairs - pred_pairs)
    total_pairs = pred_lengths[0] * pred_lengths[1]
    true_negative = max(
        total_pairs - true_positive - false_positive - false_negative, 0
    )

    return _PairConfusion(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
    )


def extract_aligned_pairs(alignment: Alignment) -> Set[Tuple[int, int]]:
    """Return the set of residue index pairs represented by the alignment.

    Indexing is zero-based with respect to the unaligned (gap-free) sequences.
    Gapped positions are ignored.
    """
    seq_x, seq_y = _ensure_pairwise_alignment(alignment)
    idx_x = idx_y = 0
    pairs: Set[Tuple[int, int]] = set()

    for residue_x, residue_y in zip(seq_x.residues, seq_y.residues):
        pos_x = idx_x if residue_x not in GAP_CHARACTERS else None
        pos_y = idx_y if residue_y not in GAP_CHARACTERS else None

        if residue_x not in GAP_CHARACTERS:
            idx_x += 1
        if residue_y not in GAP_CHARACTERS:
            idx_y += 1

        if pos_x is not None and pos_y is not None:
            pairs.add((pos_x, pos_y))

    return pairs


def extract_columns(alignment: Alignment) -> List[Tuple[str, str]]:
    """Return the alignment columns as pairs of residues (gaps retained)."""
    seq_x, seq_y = _ensure_pairwise_alignment(alignment)
    return list(zip(seq_x.residues, seq_y.residues))


def precision(predicted: Alignment, gold: Alignment) -> float:
    """Compute precision for aligned residue pairs."""
    confusion = _pair_confusion_matrix(predicted, gold)
    return _safe_divide(
        confusion.true_positive, confusion.true_positive + confusion.false_positive
    )


def recall(predicted: Alignment, gold: Alignment) -> float:
    """Compute recall for aligned residue pairs."""
    confusion = _pair_confusion_matrix(predicted, gold)
    return _safe_divide(
        confusion.true_positive, confusion.true_positive + confusion.false_negative
    )


def f1_score(predicted: Alignment, gold: Alignment) -> float:
    """Compute the harmonic mean of precision and recall."""
    prec = precision(predicted, gold)
    rec = recall(predicted, gold)
    denominator = prec + rec
    return _safe_divide(2 * prec * rec, denominator)


def sensitivity(predicted: Alignment, gold: Alignment) -> float:
    """Sensitivity is equivalent to recall for aligned residue pairs."""
    return recall(predicted, gold)


def specificity(predicted: Alignment, gold: Alignment) -> float:
    """Compute specificity for aligned residue pairs."""
    confusion = _pair_confusion_matrix(predicted, gold)
    return _safe_divide(
        confusion.true_negative, confusion.true_negative + confusion.false_positive
    )


def column_identity(predicted: Alignment, gold: Alignment) -> float:
    """Fraction of columns that exactly match between predicted and gold alignments."""
    predicted_columns = extract_columns(predicted)
    gold_columns = extract_columns(gold)

    max_len = max(len(predicted_columns), len(gold_columns))
    if max_len == 0:
        return 1.0

    matches = sum(
        1
        for predicted_column, gold_column in zip_longest(
            predicted_columns, gold_columns, fillvalue=None
        )
        if predicted_column == gold_column
    )
    return matches / max_len


__all__ = [
    "MetricFunction",
    "extract_aligned_pairs",
    "extract_columns",
    "precision",
    "recall",
    "f1_score",
    "sensitivity",
    "specificity",
    "column_identity",
]
