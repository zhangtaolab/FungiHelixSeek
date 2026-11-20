"""Vendored Viterbi/interval utilities ported from PlantHelixSeek.

This module is self-contained inside ``plant_decoder``.  It uses only numpy,
numba, and pandas and never imports a sibling PlantHelixSeek checkout.

Ported functions preserve Plant's half-open interval convention; conversion to
1-based inclusive GFF3 coordinates happens at the writer/API boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["intergenic", "intron", "five_prime_utr", "cds", "three_prime_utr"]
N_FEATURES = 5
N_LABELS = 17

CDS_BILUO_IDX = [1, 2, 3, 4]
INTRON_BILUO_IDX = [5, 6, 7, 8]
UTR5_BILUO_IDX = [9, 10, 11, 12]
UTR3_BILUO_IDX = [13, 14, 15, 16]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_SIBLING_NPZ = Path(__file__).resolve().parent / "transition_probs.npz"
_NPZ_PATH = _SIBLING_NPZ if _SIBLING_NPZ.is_file() else _DATA_DIR / "transition_probs.npz"
_npz = np.load(_NPZ_PATH)
TM_PLANT_FULL = _npz["plant_full"]
TM_PLANT_PARTIAL = _npz["plant_partial"]
TM_ANIMAL_FULL = _npz["animal_full"]
TM_ANIMAL_PARTIAL = _npz["animal_partial"]

INTERVAL_ENTITY_CLASSES = [
    [1, 2, 3, 4],
    [2, 3, 4],
    [1],
    [2],
    [3],
    [4],
]

INTERVAL_ENTITY_NAMES = [
    "transcript",
    "exon",
    "intron",
    "five_prime_utr",
    "cds",
    "three_prime_utr",
]


def token_transition_probs(
    remove_incomplete_features: bool = True, domain: str = "plant"
) -> pd.DataFrame:
    """Return the 5x5 gene feature transition probability matrix.

    Parameters
    ----------
    remove_incomplete_features : bool
        If True, use the complete-transcript matrix (requires 5'UTR + CDS + 3'UTR).
        If False, allow partial transcripts.
    domain : str
        "plant" or "animal".

    Returns
    -------
    pd.DataFrame
        5x5 DataFrame with FEATURE_NAMES as both index and columns.
        Rows sum to 1. Element (i, j) = P(next_state=j | current_state=i).
    """
    if domain == "animal":
        probs = (
            TM_ANIMAL_FULL
            if remove_incomplete_features
            else TM_ANIMAL_PARTIAL
        )
    elif domain == "plant":
        probs = (
            TM_PLANT_FULL
            if remove_incomplete_features
            else TM_PLANT_PARTIAL
        )
    else:
        raise ValueError(
            f"Unsupported domain {domain!r}. Expected 'plant' or 'animal'."
        )

    return pd.DataFrame(data=probs, index=FEATURE_NAMES, columns=FEATURE_NAMES)


def regularize_transition_matrix(
    transition_matrix: npt.ArrayLike, alpha: float
) -> np.ndarray:
    """Add *alpha* to all transitions and renormalize rows to sum to 1.

    Parameters
    ----------
    transition_matrix : array-like of shape (N, N)
        Original transition matrix.
    alpha : float
        Amount to add to all transitions (0 <= alpha <= 1).

    Returns
    -------
    np.ndarray
        Regularized transition matrix of same shape.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"Alpha must be between 0 and 1; got {alpha}")
    tm = np.asarray(transition_matrix, dtype=float)
    if tm.ndim != 2:
        raise ValueError(f"Transition matrix must be 2D; got shape {tm.shape}")
    tm = tm + alpha
    tm = tm / tm.sum(axis=1, keepdims=True)
    assert np.allclose(tm.sum(axis=1), 1)
    return tm


def transition_matrix_stationary_distribution(
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """Compute the stationary distribution of a Markov chain transition matrix.

    Parameters
    ----------
    transition_matrix : np.ndarray of shape (N, N)
        Row-stochastic transition matrix.

    Returns
    -------
    np.ndarray of shape (N,)
        Stationary distribution (probabilities sum to 1).
    """
    eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvals - 1.0))
    dist = np.real(eigenvecs[:, idx])
    dist = dist / dist.sum()
    return dist


def _validate_decode_inputs(
    emission_probs: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    initial_probs: npt.ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and prepare inputs for Viterbi decoding."""
    emission_probs = np.asarray(emission_probs, dtype=float)
    transition_matrix = np.asarray(transition_matrix, dtype=float)

    _T, N = emission_probs.shape

    if transition_matrix.shape != (N, N):
        raise ValueError(
            f"Transition matrix shape {transition_matrix.shape} doesn't match "
            f"emission probabilities states {N}"
        )

    row_sums = transition_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        if (
            np.all(np.isfinite(row_sums))
            and np.all(row_sums > 0)
            and np.allclose(row_sums, 1.0, atol=1e-6)
        ):
            logger.warning(
                "Transition matrix rows do not exactly sum to 1; normalizing rows"
            )
            transition_matrix = transition_matrix / row_sums[:, None]
        else:
            raise ValueError("Transition matrix rows must sum to 1")

    if initial_probs is None:
        initial = transition_matrix_stationary_distribution(transition_matrix)
    else:
        initial = np.asarray(initial_probs, dtype=float)
        if initial.shape != (N,):
            raise ValueError(
                f"Initial probabilities shape {initial.shape} "
                f"doesn't match number of states {N}"
            )
    if not np.isclose(initial.sum(), 1.0):
        raise ValueError(f"Initial probabilities must sum to 1; {initial=}")

    return emission_probs, transition_matrix, initial


@njit
def _viterbi_decode(
    log_emission: np.ndarray,
    log_transition: np.ndarray,
    log_initial: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated Viterbi algorithm.

    Parameters
    ----------
    log_emission : shape (T, N)
        Log emission probabilities.
    log_transition : shape (N, N)
        Log transition probabilities.
    log_initial : shape (N,)
        Log initial state probabilities.

    Returns
    -------
    np.ndarray of shape (T,) dtype=int64
        Most likely state sequence.
    """
    T, N = log_emission.shape

    viterbi_prob = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=np.int64)

    viterbi_prob[0] = log_initial + log_emission[0]

    for t in range(1, T):
        for j in range(N):
            probs = viterbi_prob[t - 1] + log_transition[:, j]
            backpointer[t, j] = np.argmax(probs)
            viterbi_prob[t, j] = probs[backpointer[t, j]] + log_emission[t, j]

    path = np.zeros(T, dtype=np.int64)
    path[T - 1] = np.argmax(viterbi_prob[T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path


def viterbi_decode(
    emission_probs: npt.ArrayLike,
    transition_matrix: npt.ArrayLike,
    initial_probs: npt.ArrayLike | None = None,
    alpha: float = 0.0,
    epsilon: float | None = None,
) -> npt.NDArray[np.int64]:
    """Find the most likely state sequence using the Viterbi algorithm.

    Parameters
    ----------
    emission_probs : array-like of shape (T, N)
        Per-position emission probabilities. Element (t, n) = P(obs at t | state n).
        Rows should be valid probability distributions (roughly sum to 1).
    transition_matrix : array-like of shape (N, N)
        State transition matrix. Rows must sum to 1.
    initial_probs : array-like of shape (N,), optional
        Initial state probabilities. If None, uses the stationary distribution
        of the transition matrix.
    alpha : float, default 0
        Transition matrix regularization amount (0 <= alpha <= 1).
        Larger values allow more transitions between states.
    epsilon : float, optional
        Minimum probability floor for numerical stability.
        Defaults to ``np.finfo(float).eps``.

    Returns
    -------
    np.ndarray of shape (T,) dtype=int64
        Most likely state index at each position.
    """
    if alpha > 0:
        transition_matrix = regularize_transition_matrix(transition_matrix, alpha)
    if epsilon is None:
        epsilon = cast(float, np.finfo(float).eps)
    if not 0 <= epsilon <= 1:
        raise ValueError(f"Epsilon must be between 0 and 1; got {epsilon}")

    emission_probs, transition_matrix, initial_probs = _validate_decode_inputs(
        emission_probs, transition_matrix, initial_probs
    )

    log_emission = np.log(np.maximum(emission_probs, epsilon))
    log_transition = np.log(np.maximum(transition_matrix, epsilon))
    log_initial = np.log(np.maximum(initial_probs, epsilon))

    return _viterbi_decode(log_emission, log_transition, log_initial)


def find_intervals(
    labels: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Find contiguous intervals of 1s in a binary label array.

    Parameters
    ----------
    labels : shape (N,) of {0, 1}
        Binary sequence labels.
    mask : shape (N,) of bool, optional
        If provided, intervals containing any masked position are discarded.

    Returns
    -------
    np.ndarray of shape (2, M)
        Start (row 0) and stop (row 1) indices, 0-based inclusive on both ends.
    """
    if labels.ndim != 1:
        raise ValueError("Labels must be a 1D array")
    if len(np.setdiff1d(np.unique(labels), [0, 1])) > 0:
        raise ValueError("Labels must be 0 or 1")

    if mask is not None:
        if mask.shape != labels.shape:
            raise ValueError(
                f"Mask shape {mask.shape} != labels shape {labels.shape}"
            )
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError(f"Mask must be boolean, got {mask.dtype}")

    diffs = np.diff(np.pad(labels, (1, 1)))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0] - 1

    if mask is not None and len(starts) > 0:
        valid = np.array(
            [np.all(mask[s : e + 1]) for s, e in zip(starts, stops)]
        )
        starts, stops = starts[valid], stops[valid]

    return np.array([starts, stops])


def find_group_intervals(
    labels: np.ndarray,
    class_groups: list[list[int]],
    mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Find intervals for groups of class labels.

    Parameters
    ----------
    labels : shape (N,) of int
        Sequence labels with values in {0, 1, 2, ...}.
    class_groups : list of list of int
        Each element lists the class indices to collapse into one output class.
    mask : shape (N,) of bool, optional
        Mask for valid positions.

    Returns
    -------
    list of np.ndarray
        One (2, M_i) array per group, with start/stop indices.
    """
    input_classes = np.unique([c for group in class_groups for c in group])
    if len(input_classes) == 0:
        return []
    if not np.issubdtype(input_classes.dtype, np.integer):
        raise ValueError("class_groups must contain only integer values")
    if (input_classes <= 0).any():
        raise ValueError("class_groups must contain only positive integers")

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be integer, got {labels.dtype}")

    result = []
    for group in class_groups:
        group_mask = np.isin(labels, group).astype(np.int8)
        group_intervals = find_intervals(group_mask, mask)
        result.append(group_intervals)

    return result


def convert_entity_labels_to_intervals(
    labels: np.ndarray,
    class_groups: list[list[int]],
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert entity label array to a structured interval DataFrame.

    Parameters
    ----------
    labels : shape (N,) of int
        Entity class indices (1-based, 0=background).
    class_groups : list of list of int
        Mappings from input classes to output entities.
    mask : shape (N,) of bool, optional
        Valid position mask.

    Returns
    -------
    pd.DataFrame
        Columns: entity (1-based int), start (0-based inclusive),
        stop (0-based inclusive).
    """
    intervals = find_group_intervals(labels, class_groups=class_groups, mask=mask)
    result = []
    for entity_idx in range(len(intervals)):
        starts, stops = intervals[entity_idx]
        for s, e in zip(starts.tolist(), stops.tolist()):
            result.append(
                {
                    "entity": int(entity_idx + 1),
                    "start": int(s),
                    "stop": int(e),
                }
            )
    result = pd.DataFrame(result, columns=["entity", "start", "stop"])
    assert not result.duplicated().any()
    return result


def convert_bilou_to_entity_labels(labels: np.ndarray) -> np.ndarray:
    """Convert 17-class BILUO labels to 5-class entity labels.

    BILUO layout: 0=intergenic, 1-4=CDS, 5-8=intron, 9-12=5'UTR, 13-16=3'UTR

    Entity mapping:
        0 -> 0 (intergenic)
        1-4 -> 3 (CDS)
        5-8 -> 1 (intron)
        9-12 -> 2 (five_prime_utr)
        13-16 -> 4 (three_prime_utr)

    Parameters
    ----------
    labels : shape (N,) int
        17-class BILUO labels.

    Returns
    -------
    np.ndarray of shape (N,) int
        5-class entity labels (0=intergenic, 1=intron, 2=5utr, 3=cds, 4=3utr).
    """
    labels = np.asarray(labels, dtype=int)
    result = np.zeros_like(labels)
    result[(labels >= 1) & (labels <= 4)] = 3
    result[(labels >= 5) & (labels <= 8)] = 1
    result[(labels >= 9) & (labels <= 12)] = 2
    result[(labels >= 13) & (labels <= 16)] = 4
    return result


def aggregate_logits_np(
    probs_17: np.ndarray,
) -> np.ndarray:
    """Aggregate 17-class BILUO probabilities to 5-class feature probabilities.

    BILUO layout: 0=intergenic, 1-4=CDS, 5-8=intron, 9-12=5UTR, 13-16=3UTR
    Feature order: [intergenic, intron, 5'UTR, CDS, 3'UTR]
    """
    probs = np.asarray(probs_17, dtype=float)
    result = np.zeros(probs.shape[:-1] + (5,), dtype=float)
    result[..., 0] = probs[..., 0]
    result[..., 1] = probs[..., 5:9].sum(axis=-1)
    result[..., 2] = probs[..., 9:13].sum(axis=-1)
    result[..., 3] = probs[..., 1:5].sum(axis=-1)
    result[..., 4] = probs[..., 13:17].sum(axis=-1)
    return result


def aggregate_probs17_to_features(probs17: np.ndarray) -> np.ndarray:
    """Aggregate ``(..., 17)`` probabilities into float32 ``(..., 5)`` features."""
    probs = np.asarray(probs17, dtype=np.float64)
    if probs.ndim < 1 or probs.shape[-1] != N_LABELS:
        raise ValueError(
            f"Expected last dimension {N_LABELS}, got shape {probs.shape}"
        )
    return np.asarray(aggregate_logits_np(probs), dtype=np.float32)
