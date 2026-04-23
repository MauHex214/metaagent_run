"""Pass 0 — Corpus-driven head-word induction for Step 3.

Given a list of `Decomposed` records (from rules.decompose), build the
head-word set H as follows:

    1. Tokenize each root on underscores (drop empty fragments).
    2. For each token t, compute document frequency DF(t) = #{keys whose root
       contains t as a token}.
    3. Choose the smallest DF threshold τ such that the head-word set
       H = {t : DF(t) ≥ τ} covers ≥ `coverage_target` of the keys, where a
       key is "covered" iff at least one of its root tokens lies in H.

The induced H and threshold τ are both data-driven; no external vocabulary
(MIxS, domain glossary, etc.) is consulted.

Multi-coverage sensitivity: `induce_headwords` optionally accepts a list of
coverage targets and returns (τ, |H|, orphan_rate) for each — used by
diagnose.py for the Method's parameter-sensitivity table.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

try:
    from .rules import Decomposed, root_tokens
except ImportError:
    # Allow running as a standalone script in the same dir
    from rules import Decomposed, root_tokens  # type: ignore


# ═══════════════════════════════════════════════════════════════════
#  Core DF statistics
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TokenStats:
    df: Dict[str, int]                 # token → document frequency
    total_keys: int                    # total number of decomposed entries
    token_to_keys: Dict[str, Set[str]] # token → set of root strings

    @property
    def total_tokens(self) -> int:
        return len(self.df)


def compute_token_stats(decomposed: List[Decomposed]) -> TokenStats:
    """Count document frequency over root tokens.

    A key's root contributes each of its DISTINCT tokens exactly once
    to the DF of that token (standard "document frequency" definition).
    """
    df: Counter = Counter()
    token_to_keys: Dict[str, Set[str]] = defaultdict(set)
    for d in decomposed:
        toks = set(root_tokens(d))
        if not toks:
            continue
        for t in toks:
            df[t] += 1
            token_to_keys[t].add(d.root)
    return TokenStats(
        df=dict(df),
        total_keys=len(decomposed),
        token_to_keys=dict(token_to_keys),
    )


# ═══════════════════════════════════════════════════════════════════
#  Threshold solver
# ═══════════════════════════════════════════════════════════════════

def _coverage_for_threshold(
    decomposed: List[Decomposed], stats: TokenStats, tau: int,
) -> Tuple[float, int, Set[str]]:
    """Given a candidate DF threshold τ, return (coverage, |H|, H).

    coverage = fraction of keys that have at least one root token with DF ≥ τ.
    """
    H = {t for t, df in stats.df.items() if df >= tau}
    if not H:
        return 0.0, 0, H
    covered = 0
    for d in decomposed:
        toks = root_tokens(d)
        if any(t in H for t in toks):
            covered += 1
    coverage = covered / max(stats.total_keys, 1)
    return coverage, len(H), H


def induce_headwords(
    decomposed: List[Decomposed],
    coverage_target: float = 0.95,
    stats: TokenStats | None = None,
) -> Tuple[Set[str], int, float]:
    """Solve for the smallest τ such that coverage ≥ coverage_target.

    Returns (H, τ, coverage_achieved). If the target is unachievable
    (e.g., corpus has too many singletons), returns the best attainable
    set using τ=1.
    """
    stats = stats or compute_token_stats(decomposed)
    if not stats.df:
        return set(), 0, 0.0

    # Binary search over candidate τ values. Candidate τ's are distinct
    # observed DF values — no need to try τ between them.
    distinct_dfs = sorted(set(stats.df.values()))  # ascending

    # Higher τ ⇒ smaller H ⇒ lower coverage. So to find the SMALLEST τ
    # that still meets coverage_target, we want the LARGEST τ whose
    # coverage ≥ target — actually we want smallest H, which is largest τ.
    # Smaller τ gives larger H and higher coverage, but we want the
    # MINIMAL H that still satisfies coverage — equivalently, the
    # largest τ where coverage still ≥ target.
    best_tau = distinct_dfs[0]
    best_H: Set[str] = set()
    best_coverage = 0.0

    lo, hi = 0, len(distinct_dfs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        tau = distinct_dfs[mid]
        coverage, _, H = _coverage_for_threshold(decomposed, stats, tau)
        if coverage >= coverage_target:
            best_tau, best_H, best_coverage = tau, H, coverage
            lo = mid + 1  # try even larger τ
        else:
            hi = mid - 1

    if not best_H:
        # Target unattainable even at τ=1. Return τ=1 result.
        tau1 = distinct_dfs[0]
        coverage, _, H = _coverage_for_threshold(decomposed, stats, tau1)
        return H, tau1, coverage

    return best_H, best_tau, best_coverage


# ═══════════════════════════════════════════════════════════════════
#  Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════

def sensitivity_table(
    decomposed: List[Decomposed],
    coverage_targets: Iterable[float] = (0.85, 0.88, 0.90, 0.92, 0.95),
    stats: TokenStats | None = None,
) -> List[Dict[str, float]]:
    """For each target coverage, report (τ, |H|, achieved_coverage, orphan_rate)."""
    stats = stats or compute_token_stats(decomposed)
    rows: List[Dict[str, float]] = []
    for target in coverage_targets:
        H, tau, cov = induce_headwords(decomposed, target, stats=stats)
        orphan_rate = 1.0 - cov
        rows.append({
            "coverage_target": float(target),
            "tau": int(tau),
            "H_size": len(H),
            "coverage_achieved": float(cov),
            "orphan_rate": float(orphan_rate),
        })
    return rows


def tau_coverage_curve(
    decomposed: List[Decomposed],
    stats: TokenStats | None = None,
    taus: Iterable[int] = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100),
) -> List[Dict[str, float]]:
    """Direct enumeration: for each τ, report |H(τ)| and coverage.

    Useful for plotting the actual shape of the coverage curve —
    the sensitivity table may pick degenerate τ values when target is
    unreachable in the discrete DF grid.
    """
    stats = stats or compute_token_stats(decomposed)
    rows: List[Dict[str, float]] = []
    for tau in taus:
        coverage, hsize, _ = _coverage_for_threshold(decomposed, stats, tau)
        rows.append({
            "tau": int(tau),
            "H_size": hsize,
            "coverage": float(coverage),
            "orphan_rate": float(1.0 - coverage),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════
#  Long-tail descriptors — for the Method's parameter-diagnosis narrative
# ═══════════════════════════════════════════════════════════════════

def df_histogram(stats: TokenStats, bins: Iterable[Tuple[int, int]] | None = None) -> Dict[str, int]:
    """Bucket token count by DF ranges.

    `bins` is a list of (lo, hi) inclusive-inclusive ranges. If None,
    uses a default set that captures Zipf-style long tails.
    """
    if bins is None:
        bins = [
            (1, 1),
            (2, 2),
            (3, 5),
            (6, 10),
            (11, 20),
            (21, 50),
            (51, 100),
            (101, 500),
            (501, 10_000_000),
        ]
    out: Dict[str, int] = {}
    for lo, hi in bins:
        label = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 10_000_000 else f">={lo}"
        out[label] = sum(1 for df in stats.df.values() if lo <= df <= hi)
    return out


def zipf_points(stats: TokenStats, max_rank: int = 500) -> List[Tuple[int, int]]:
    """Return (rank, DF) pairs, sorted by DF descending, for log-log plot."""
    df_sorted = sorted(stats.df.values(), reverse=True)[:max_rank]
    return [(i + 1, df) for i, df in enumerate(df_sorted)]


def non_headword_summary(
    stats: TokenStats, headwords: Set[str],
) -> Dict[str, int]:
    """Count how the non-headword tail is distributed."""
    non_H = [df for t, df in stats.df.items() if t not in headwords]
    return {
        "non_headword_tokens": len(non_H),
        "df1_tokens": sum(1 for df in non_H if df == 1),
        "df2_tokens": sum(1 for df in non_H if df == 2),
        "df3_5_tokens": sum(1 for df in non_H if 3 <= df <= 5),
        "df6_10_tokens": sum(1 for df in non_H if 6 <= df <= 10),
        "df_gt_10_tokens": sum(1 for df in non_H if df > 10),
    }


def orphan_keys(
    decomposed: List[Decomposed], headwords: Set[str],
) -> List[Decomposed]:
    """Keys whose root contains no head-word token."""
    out: List[Decomposed] = []
    for d in decomposed:
        toks = root_tokens(d)
        if not toks:
            out.append(d)
            continue
        if not any(t in headwords for t in toks):
            out.append(d)
    return out


def orphan_buckets_by_token_count(
    orphans: List[Decomposed],
) -> Dict[str, int]:
    """Split orphans by how many tokens their root has (hints at shape)."""
    buckets: Counter = Counter()
    for d in orphans:
        n = len(root_tokens(d))
        key = f"{n}_token" if n <= 5 else ">=6_tokens"
        buckets[key] += 1
    return dict(buckets)


# ═══════════════════════════════════════════════════════════════════
#  Top-DF tokens — for inspection
# ═══════════════════════════════════════════════════════════════════

def top_tokens(stats: TokenStats, k: int = 50) -> List[Tuple[str, int]]:
    return sorted(stats.df.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


if __name__ == "__main__":
    from rules import decompose

    sample_keys = [
        "depth", "water_depth", "sampling_depth", "in_situ_depth",
        "water_temperature", "sea_surface_temperature", "temperature",
        "salinity", "in_situ_salinity", "water_salinity",
        "ph", "water_ph", "sample_ph",
        "ammonia", "ammonium", "nitrate", "nitrite",
        "chlorophyll_a", "chlorophyll",
        "137cs", "14c_age", "13c",
        "area", "area_ha", "area_km2",
        "some_uniquely_weird_orphan_key",
    ]
    decomp = [decompose(k) for k in sample_keys]
    stats = compute_token_stats(decomp)
    print(f"Total keys: {stats.total_keys}, total tokens: {stats.total_tokens}")
    print(f"Top 10 tokens: {top_tokens(stats, 10)}")
    print(f"DF histogram: {df_histogram(stats)}")
    print(f"Sensitivity: {sensitivity_table(decomp, (0.5, 0.7, 0.9))}")
