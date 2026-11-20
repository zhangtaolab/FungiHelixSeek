"""Prior resolution for the vendored Plant decoder.

Priors are passed explicitly into decoding; no module globals are monkeypatched.
The fungal matrix/initial distribution come from the repository NPZ, while the
Plant matrix uses Plant's convention of deriving its initial state from the
matrix's stationary distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .viterbi_utils import (
    FEATURE_NAMES,
    N_FEATURES,
    TM_PLANT_FULL,
    transition_matrix_stationary_distribution,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FUNGAL_TRANSITION_NPZ = (
    _REPO_ROOT / "data" / "gene_annotation" / "transition_probs_fungi.npz"
)


def _validate_priors(
    transition: np.ndarray,
    initial: np.ndarray,
    *,
    transition_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = f" in {transition_path}" if transition_path is not None else ""
    if transition.shape != (N_FEATURES, N_FEATURES):
        raise ValueError(
            f"Expected a {N_FEATURES}x{N_FEATURES} transition matrix{source}, "
            f"got {transition.shape}"
        )
    if initial.shape != (N_FEATURES,):
        raise ValueError(
            f"Expected {N_FEATURES} initial probabilities{source}, "
            f"got {initial.shape}"
        )
    return transition, initial


def load_fungal_priors(
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the fungal 5x5 transition matrix and initial distribution.

    Parameters
    ----------
    path : str | Path, optional
        Explicit NPZ path.  Defaults to
        ``data/gene_annotation/transition_probs_fungi.npz`` in the repository.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(transition_probs, initial_probs)`` with shapes ``(5, 5)`` and
        ``(5,)`` in ``FEATURE_NAMES`` order.
    """
    transition_path = (
        Path(path) if path is not None else DEFAULT_FUNGAL_TRANSITION_NPZ
    )
    with np.load(transition_path) as data:
        missing = [
            key
            for key in ("transition_probs", "initial_probs")
            if key not in data.files
        ]
        if missing:
            raise KeyError(
                f"{transition_path} is missing priors arrays {missing}; "
                f"found keys: {list(data.files)}"
            )
        transition = np.asarray(data["transition_probs"], dtype=np.float64)
        initial = np.asarray(data["initial_probs"], dtype=np.float64)
    return _validate_priors(transition, initial, transition_path=transition_path)


def resolve_priors(
    prior: str | np.ndarray = "fungi",
    transition_npz: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a prior selector into an explicit transition matrix + initial.

    Parameters
    ----------
    prior : str | np.ndarray
        ``"fungi"`` loads the repository fungal priors (with the fungal
        initial distribution); ``"plant"`` uses the vendored Plant full matrix
        and derives its initial distribution from the matrix's stationary
        distribution.  A ``(5, 5)`` array-like is also accepted, with its
        stationary distribution used as the initial.
    transition_npz : str | Path, optional
        Explicit fungal-format NPZ path.  Takes precedence over *prior*.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(transition_probs, initial_probs)`` in ``FEATURE_NAMES`` order.
    """
    if transition_npz is not None:
        return load_fungal_priors(transition_npz)

    if isinstance(prior, str):
        key = prior.strip().lower()
        if key in {"fungi", "fungal"}:
            return load_fungal_priors()
        if key == "plant":
            transition = np.asarray(TM_PLANT_FULL, dtype=np.float64)
            initial = transition_matrix_stationary_distribution(transition)
            return _validate_priors(transition, initial)
        raise ValueError(
            f"Unsupported prior {prior!r}. Expected 'fungi', 'plant', or a "
            f"{N_FEATURES}x{N_FEATURES} transition matrix."
        )

    transition = np.asarray(prior, dtype=np.float64)
    if transition.shape != (N_FEATURES, N_FEATURES):
        raise ValueError(
            f"Expected a {N_FEATURES}x{N_FEATURES} transition matrix, "
            f"got {transition.shape}"
        )
    initial = transition_matrix_stationary_distribution(transition)
    return _validate_priors(transition, initial)


__all__ = [
    "DEFAULT_FUNGAL_TRANSITION_NPZ",
    "FEATURE_NAMES",
    "N_FEATURES",
    "load_fungal_priors",
    "resolve_priors",
]
