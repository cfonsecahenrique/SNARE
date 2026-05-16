"""
Shared utilities for the plotting scripts that read from ``outputs/results.csv``.

Exposes:
    * Paths and the standard "public-information" filter parameters.
    * The 4-bit second-order norm naming table (NORM_MAPPING) and the
      Leading Four list.
    * Helpers for decoding EBSNs (flatten, base-norm identification,
      cooperative / competitive half-rule labels).
    * load_data()         -- load + merge results.csv + 8bit-norms metadata.
    * filter_standard()   -- the standard public-info parameter filter,
                             plus labelling of the two half-rules.
    * canonical_filter()  -- restricts a frame to the paper-Eq.1 canonical
                             EBSNs (or keeps baseline gamma=0 rows).
"""

from __future__ import annotations

import ast
import sys
from itertools import chain
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOTTING_DIR = Path(__file__).resolve().parent
SNARE_ROOT = PLOTTING_DIR.parent
PLOTS_DIR = PLOTTING_DIR / "plots"
RESULTS_CSV = SNARE_ROOT / "outputs" / "results.csv"
NORMS_CSV = SNARE_ROOT / "data" / "all_8bit_norms_with_dnf.csv"

# Make sure SNARE root is importable for canonical_ebsns.py
if str(SNARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SNARE_ROOT))

from canonical_ebsns import canonical_ebsns  # noqa: E402

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
NORM_MAPPING = {
    (0, 0, 0, 0): "All Bad",
    (0, 0, 0, 1): "Shunning",
    (0, 0, 1, 0): "U2",
    (0, 0, 1, 1): "Image Scoring",
    (0, 1, 0, 0): "U4",
    (0, 1, 0, 1): "U5",
    (0, 1, 1, 0): "U6",
    (0, 1, 1, 1): "U7",
    (1, 0, 0, 0): "paradoxical Simple Standing",
    (1, 0, 0, 1): "Stern Judging",
    (1, 0, 1, 0): "U10",
    (1, 0, 1, 1): "Simple Standing",
    (1, 1, 0, 0): "paradoxical Image Scoring",
    (1, 1, 0, 1): "paradoxical Shunning",
    (1, 1, 1, 0): "U14",
    (1, 1, 1, 1): "All Good",
}

LEADING_FOUR = ["Stern Judging", "Simple Standing", "Shunning", "Image Scoring"]

# Stable colour assignment for named 2nd-order norms (used across plots)
NORM_COLOURS = {
    "Stern Judging": "#1f77b4",
    "Simple Standing": "#ff7f0e",
    "Shunning": "#2ca02c",
    "Image Scoring": "#bcbd22",
    "All Good": "#9467bd",
    "All Bad": "#d62728",
    "paradoxical Simple Standing": "#8c564b",
    "paradoxical Image Scoring": "#17becf",
    "paradoxical Shunning": "#e377c2",
    "U2": "#aec7e8",
    "U4": "#c7c7c7",
    "U5": "#ffbb78",
    "U6": "#98df8a",
    "U7": "#c5b0d5",
    "U10": "#dbdb8d",
    "U14": "#f7b6d2",
    "Other": "#7f7f7f",
    "Unknown": "#cccccc",
}

# Default filter values (public-information regime used throughout the paper)
DEFAULT_FILTERS = dict(
    z_min=40,
    q=1.0,
    gens_min=400,
    chi=0.01,
    eps=0.01,
    target_gammas=(0, 0.5, 1),
)


# ---------------------------------------------------------------------------
# Helpers for decoding EBSNs
# ---------------------------------------------------------------------------
def flatten_ebsn_to_str(ebsn) -> str:
    """Flatten the nested ``[[(a,b),(c,d)],[(e,f),(g,h)]]`` 8-bit list to an 8-char string."""
    if isinstance(ebsn, str):
        ebsn = ast.literal_eval(ebsn)
    flat = list(chain.from_iterable(chain.from_iterable(ebsn)))
    return "".join(str(int(b)) for b in flat)


def flatten_base_sn_to_str(base_sn) -> str:
    if isinstance(base_sn, str):
        base_sn = ast.literal_eval(base_sn)
    return "".join(str(int(b)) for b in chain.from_iterable(base_sn))


def identify_base_norm(base_norm_str: str) -> str:
    try:
        norm = ast.literal_eval(base_norm_str)
    except Exception:
        return "Unknown"
    flat = [int(x) for pair in norm for x in pair]
    return NORM_MAPPING.get(tuple(flat), "Unknown")


def identify_norm_from_str(s: str) -> str:
    if pd.isna(s) or len(s) < 4:
        return "Unknown"
    vec = tuple(int(b) for b in s)
    return NORM_MAPPING.get(vec, "Other")


def get_emotional_norms(row: pd.Series) -> Tuple[str, str]:
    """Return ``(cooperative-rule-name, competitive-rule-name)`` for one row.

    At ``gamma=0`` (baseline) both halves reduce to the base norm itself.
    """
    if row["gamma_center"] == 0:
        base_name = identify_norm_from_str(row["4bit_orig"])
        return base_name, base_name

    ebsn = row["8bit_vector"]
    if pd.isna(ebsn) or len(ebsn) < 8:
        return "Unknown", "Unknown"

    # Bit positions: 0,2,4,6 = Competitive (M); 1,3,5,7 = Cooperative (O)
    comp_vec = (int(ebsn[0]), int(ebsn[2]), int(ebsn[4]), int(ebsn[6]))
    coop_vec = (int(ebsn[1]), int(ebsn[3]), int(ebsn[5]), int(ebsn[7]))
    return (
        NORM_MAPPING.get(coop_vec, "Other"),
        NORM_MAPPING.get(comp_vec, "Other"),
    )


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def load_data(
    results_csv: Path = RESULTS_CSV,
    norms_csv: Path = NORMS_CSV,
) -> pd.DataFrame:
    """Load and merge ``results.csv`` with the 8-bit-norm metadata."""
    results_df = pd.read_csv(results_csv, on_bad_lines="skip", engine="python")
    norms_df = pd.read_csv(norms_csv, dtype={"8bit_vector": str})

    results_df["8bit_vector"] = results_df["eb_social_norm"].apply(flatten_ebsn_to_str)
    results_df["4bit_orig"] = (
        results_df["base_social_norm"].apply(eval).apply(flatten_base_sn_to_str)
    )

    merged = pd.merge(
        results_df,
        norms_df[["8bit_vector", "Emotion_Leniency", "DNF", "DNF_literals"]],
        on=["8bit_vector"],
        how="left",
    )
    merged["DNF_literals"] = pd.to_numeric(merged["DNF_literals"], errors="coerce")
    merged["4bit_norm"] = merged["base_social_norm"]
    merged["base_social_norm"] = merged["base_social_norm"].apply(identify_base_norm)
    return merged


def filter_standard(
    merged: pd.DataFrame,
    *,
    z_min: int = DEFAULT_FILTERS["z_min"],
    q: float = DEFAULT_FILTERS["q"],
    gens_min: int = DEFAULT_FILTERS["gens_min"],
    chi: float = DEFAULT_FILTERS["chi"],
    eps: float = DEFAULT_FILTERS["eps"],
    target_gammas: Iterable[float] | None = DEFAULT_FILTERS["target_gammas"],
    label_halves: bool = True,
    drop_trivial: bool = True,
    normalise_acr: bool = True,
) -> pd.DataFrame:
    """Apply the standard public-info filter and (optionally) label half-rules.

    Parameters
    ----------
    target_gammas : iterable of float or None
        Restrict to these gamma values, or None to skip the gamma filter.
    label_halves : bool
        If True, adds 'Cooperative-Social Norm' / 'Competitive-Social Norm' columns.
    drop_trivial : bool
        Drop rows with ``Emotion_Leniency == 1`` (except at ``gamma_center == 0``,
        which encodes the baseline).
    normalise_acr : bool
        If True, divide ``average_cooperation`` by 100 when it looks stored as
        a percentage.
    """
    df = merged.copy()
    if target_gammas is not None:
        df = df[df["gamma_center"].isin(list(target_gammas))]
    df = df[(df.Z >= z_min) & (df.q == q) & (df.gens >= gens_min)]
    df = df[(df.chi == chi) & (df.eps == eps)]

    if label_halves:
        norms_split = df.apply(get_emotional_norms, axis=1)
        df["Cooperative-Social Norm"] = [x[0] for x in norms_split]
        df["Competitive-Social Norm"] = [x[1] for x in norms_split]

    if drop_trivial:
        df = df[(df["Emotion_Leniency"] != 1) | (df["gamma_center"] == 0)]

    if normalise_acr and df["average_cooperation"].max() > 1.1:
        df = df.copy()
        df["average_cooperation"] = df["average_cooperation"] / 100.0
    return df


def canonical_filter(df: pd.DataFrame, *, keep_baselines: bool = True) -> pd.DataFrame:
    """Restrict ``df`` to canonical EBSNs from paper Eq.1.

    If ``keep_baselines`` is True, rows with ``gamma_center == 0`` are retained
    regardless (baseline 2nd-order norms aren't in the EBSN canonical set).
    """
    canonical_str = {"".join(map(str, n)) for n in canonical_ebsns()}
    mask = df["8bit_vector"].isin(canonical_str)
    if keep_baselines and "gamma_center" in df.columns:
        mask = mask | (df["gamma_center"] == 0)
    return df[mask]
