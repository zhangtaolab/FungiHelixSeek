"""Self-contained vendored Plant decoder primitives.

This package has no dependency on a sibling PlantHelixSeek checkout.  Later
tasks add decoding, repair, writer, and API modules on top of these utilities.
"""

from __future__ import annotations

from .api import DecoderConfig, SUPPORTED_DECODERS, decode_contig
from .model import Exon, Gene
from .orf import (
    ORF_MAX_SHIFT,
    correct_all_genes,
    correct_gene_orf,
    frame_shifts,
    merge_genes_by_intron_prob,
)
from .postprocess import (
    detect_all_utrs,
    filter_low_confidence_single_exons,
    filter_to_min_cds_length,
    filter_to_min_gene_length,
    filter_to_valid_genes,
)
from .priors import DEFAULT_FUNGAL_TRANSITION_NPZ, load_fungal_priors, resolve_priors
from .repair import (
    cds_seq,
    first_stop,
    rc,
    repair_exons,
    repair_genes,
    valid_orf,
)
from .viterbi import decode_viterbi
from .writer import gene_to_gff_lines, write_gff3
from .viterbi_utils import (
    FEATURE_NAMES,
    INTERVAL_ENTITY_CLASSES,
    INTERVAL_ENTITY_NAMES,
    N_FEATURES,
    N_LABELS,
    aggregate_logits_np,
    aggregate_probs17_to_features,
    convert_bilou_to_entity_labels,
    convert_entity_labels_to_intervals,
    find_group_intervals,
    find_intervals,
    regularize_transition_matrix,
    token_transition_probs,
    transition_matrix_stationary_distribution,
    viterbi_decode,
)

__all__ = [
    "DEFAULT_FUNGAL_TRANSITION_NPZ",
    "DecoderConfig",
    "Exon",
    "FEATURE_NAMES",
    "Gene",
    "INTERVAL_ENTITY_CLASSES",
    "INTERVAL_ENTITY_NAMES",
    "N_FEATURES",
    "N_LABELS",
    "ORF_MAX_SHIFT",
    "SUPPORTED_DECODERS",
    "aggregate_logits_np",
    "aggregate_probs17_to_features",
    "cds_seq",
    "convert_bilou_to_entity_labels",
    "convert_entity_labels_to_intervals",
    "correct_all_genes",
    "correct_gene_orf",
    "decode_contig",
    "decode_viterbi",
    "detect_all_utrs",
    "first_stop",
    "filter_low_confidence_single_exons",
    "filter_to_min_cds_length",
    "filter_to_min_gene_length",
    "filter_to_valid_genes",
    "find_group_intervals",
    "find_intervals",
    "frame_shifts",
    "gene_to_gff_lines",
    "load_fungal_priors",
    "merge_genes_by_intron_prob",
    "rc",
    "regularize_transition_matrix",
    "repair_exons",
    "repair_genes",
    "resolve_priors",
    "token_transition_probs",
    "transition_matrix_stationary_distribution",
    "valid_orf",
    "viterbi_decode",
    "write_gff3",
]
