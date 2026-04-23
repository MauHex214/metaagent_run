"""Pass B preparation — concept family construction + Union-Find merge.

Given Pass A decomposition results and an induced anchor-word set H,
this module provides:

    build_families(decomposed, H)
        → Dict[anchor_word, Family]  — each key joins every family whose
           anchor appears in its root tokens.

    split_oversized_family(family, max_size)
        → List[Family]  — chunk a large family so each chunk fits in a
           single LLM call. Chunks are sorted-by-root so similar members
           cluster together.

    UnionFind merge across families, respecting hard partition_key
    constraint (isotope, substance). Two keys with the same
    partition_key may be merged by any family's LLM decision; two keys
    with different partition_keys are refused the merge regardless.

    select_canonical(group, pmid_count)
        → canonical name for a Union-Find group, by the four-tier rule:
           1. highest PMID coverage
           2. fewest tokens
           3. shortest string
           4. lexicographic

No LLM calls in this module.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from .rules import Decomposed, root_tokens
except ImportError:
    from rules import Decomposed, root_tokens  # type: ignore


LOGGER = logging.getLogger("step3.family")


# ═══════════════════════════════════════════════════════════════════
#  Family dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Family:
    """A concept family: one anchor word + the decomposed keys that
    include that anchor as one of their root tokens.

    `chunk_index` is used when a large family is split for LLM prompt
    size limits — chunks share the same anchor but are processed as
    independent LLM calls. chunk_index=0 means "not split".
    """
    anchor: str
    members: List[Decomposed]
    chunk_index: int = 0
    chunk_total: int = 1

    @property
    def id(self) -> str:
        if self.chunk_total == 1:
            return self.anchor
        return f"{self.anchor}#{self.chunk_index + 1}of{self.chunk_total}"

    @property
    def size(self) -> int:
        return len(self.members)


# ═══════════════════════════════════════════════════════════════════
#  Family construction
# ═══════════════════════════════════════════════════════════════════

def build_families(
    decomposed: List[Decomposed], headwords: Set[str],
) -> Dict[str, Family]:
    """Assign each decomposed key to every family whose anchor is in
    its root-token set. Returns {anchor: Family}.

    Keys whose root has no token in H become orphans (not in any
    family); the caller handles them separately.
    """
    fam: Dict[str, List[Decomposed]] = defaultdict(list)
    for d in decomposed:
        toks = set(root_tokens(d))
        hits = toks & headwords
        for a in hits:
            fam[a].append(d)
    return {a: Family(anchor=a, members=members) for a, members in fam.items()}


def find_orphans(
    decomposed: List[Decomposed], headwords: Set[str],
) -> List[Decomposed]:
    """Keys with no root token in H."""
    return [d for d in decomposed if not (set(root_tokens(d)) & headwords)]


def split_oversized_family(
    family: Family, max_size: int,
) -> List[Family]:
    """Split a family into ≤max_size chunks.

    Members are sorted by (partition_key, root, original) so chunk
    boundaries are deterministic and similar items stay together —
    this minimises cross-chunk misses. Note that Union-Find across
    multi-family membership still recovers many cross-chunk links
    because most keys belong to multiple families.
    """
    if family.size <= max_size:
        return [family]

    sorted_members = sorted(
        family.members,
        key=lambda d: (
            (d.partition_key[0] or "", tuple(sorted(d.partition_key[1])) if d.partition_key[1] else ()),
            d.root,
            d.original,
        ),
    )
    chunks: List[List[Decomposed]] = []
    for i in range(0, len(sorted_members), max_size):
        chunks.append(sorted_members[i : i + max_size])

    total = len(chunks)
    return [
        Family(anchor=family.anchor, members=chunk, chunk_index=i, chunk_total=total)
        for i, chunk in enumerate(chunks)
    ]


def split_all_families(
    families: Dict[str, Family], max_size: int,
) -> List[Family]:
    """Return a flat list of Family objects after splitting oversized ones."""
    out: List[Family] = []
    for f in families.values():
        out.extend(split_oversized_family(f, max_size))
    return out


def drop_singleton_families(families: Iterable[Family]) -> List[Family]:
    """LLM calls on size-1 families cannot produce any merge; skip them.

    The single member is still recorded (elsewhere) as its own canonical
    unless another family unions it with someone else.
    """
    return [f for f in families if f.size >= 2]


# ═══════════════════════════════════════════════════════════════════
#  Union-Find with partition-key constraint
# ═══════════════════════════════════════════════════════════════════

class PartitionKeyUnionFind:
    """Union-Find keyed by `decomposed.original`, refusing unions
    across different partition_keys (isotope, substance).

    union(a, b) silently rejects if a.partition_key != b.partition_key.
    This is the hard-constraint backstop: even if multiple families'
    LLMs all agree to merge across substances, the merge is refused.
    """

    def __init__(self, decomposed: List[Decomposed]):
        self._parent: Dict[str, str] = {d.original: d.original for d in decomposed}
        self._rank: Dict[str, int] = {d.original: 0 for d in decomposed}
        self._partition_key: Dict[str, Tuple] = {
            d.original: _partition_key_tuple(d) for d in decomposed
        }
        self.refused_count = 0

    def find(self, x: str) -> str:
        # Path compression
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            nxt = self._parent[x]
            self._parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self._partition_key[ra] != self._partition_key[rb]:
            self.refused_count += 1
            return False
        # Union by rank
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def groups(self) -> Dict[str, List[str]]:
        """Return {root_representative: [members...]} for all groups."""
        out: Dict[str, List[str]] = defaultdict(list)
        for key in self._parent:
            out[self.find(key)].append(key)
        return dict(out)


def _partition_key_tuple(d: Decomposed) -> Tuple:
    """Hashable form of partition_key for Union-Find storage."""
    iso, subst = d.partition_key
    return (iso, frozenset(subst) if subst else None)


def apply_family_partitions(
    uf: PartitionKeyUnionFind,
    partitions_by_family: Dict[str, List[List[str]]],
) -> None:
    """For each family, for each equivalence group that family's LLM
    returned, union all pairs within the group.

    `partitions_by_family[family_id]` is a list of groups; each group
    is a list of `original` key strings. Members of the same group
    become candidates for union; Union-Find enforces the partition-key
    constraint.
    """
    for family_id, groups in partitions_by_family.items():
        for group in groups:
            if len(group) < 2:
                continue
            pivot = group[0]
            for other in group[1:]:
                uf.union(pivot, other)


# ═══════════════════════════════════════════════════════════════════
#  Class-1 SAMPLING-CONTEXT auto-merge (deterministic safety net)
# ═══════════════════════════════════════════════════════════════════
#
# The family-internal LLM partition stage is the primary mechanism for
# applying the five qualifier-class merge/split rules from dedup_v1.txt.
# In practice, however, LLM occasionally fails to merge pairs that the
# Class-1 SAMPLING-CONTEXT rule unambiguously covers (e.g. `season` vs
# `sampling_season`; `site` vs `sample_site`), especially when a family
# is split across chunks and the two members end up in different chunks.
#
# This auto-merge is a deterministic post-LLM pass that enforces
# Class-1 rule-by-construction. For every pair of Union-Find groups
# (A, B) where tokens(A) and tokens(B) reduce to the *same* core set
# after stripping a hand-curated list of sampling-context tokens AND
# both share the same hard partition_key (isotope, substance), we
# union them.
#
# The CONTEXT set is taken verbatim from dedup_v1.txt Class-1:
#   sampling_, sample_, collection_, in_situ_, measured_,
#   environmental_, isolation_, study_, water_ (aquatic-scope).
# It encodes domain convention, not corpus statistics; it is stable
# across upstream data changes within aquatic environmental scope.
#
# Safety posture: the rule never SPLITS (it only adds union edges). It
# is also guarded against over-merging all-context canonicals by
# skipping entries whose core is empty (every token is a context
# qualifier and there is no substantive concept left to match on).

SAMPLING_CONTEXT_TOKENS: frozenset = frozenset({
    # Class-1 tokens from dedup_v1.txt, with common morphological
    # variants added to catch author-form variation observed in the
    # corpus (coll_/collect_/collecting_/sampl_/sampled_ etc).
    # sampling / sample
    "sampling", "sample", "sampled", "samp", "sampl",
    # collection / collect
    "collection", "collected", "collecting", "collect", "coll",
    "collector",
    # isolation
    "isolation", "isolated", "isolating",
    # in situ (after "in" stopword filter)
    "situ",
    # measure / measurement
    "measured", "measurement", "measure", "measuring",
    # environmental
    "environmental", "environment",
    # study
    "study", "studied",
    # water (Class-1 in aquatic context per dedup_v1.txt)
    "water", "waters",
})


def _core_tokens(d: Decomposed) -> frozenset:
    """Token set of d.root, minus stopwords (already by root_tokens),
    minus SAMPLING-CONTEXT modifiers."""
    return frozenset(root_tokens(d)) - SAMPLING_CONTEXT_TOKENS


def auto_merge_sampling_context(
    uf: PartitionKeyUnionFind,
    decomposed: List[Decomposed],
) -> Dict[str, object]:
    """Rule-based Class-1 SAMPLING-CONTEXT auto-merge applied to the
    current UF state (call AFTER apply_family_partitions, BEFORE
    canonical selection).

    Signature per UF group = the MINIMUM-size non-empty core observed
    across the group's members (most stripped-down base concept). Two
    UF groups whose signatures are identical AND whose partition_keys
    match get unioned. Using the min-core handles the case where a
    group contains one bare/context-heavy member AND one or more
    fuller-context members — the bare member is the semantic anchor
    and determines the group's identity.

    Returns a dict with counts and a merge log for auditing.
    """
    # Collect every member's (core, partition_key) for each UF root.
    root_to_pairs: Dict[str, List[Tuple[frozenset, Tuple]]] = defaultdict(list)
    root_to_rep_name: Dict[str, str] = {}  # stable name for reporting
    for d in decomposed:
        r = uf.find(d.original)
        core = _core_tokens(d)
        if core:
            root_to_pairs[r].append((core, d.partition_key))
        # Track a stable name per root for the audit log (smallest member)
        cur = root_to_rep_name.get(r)
        if cur is None or (len(d.original), d.original) < (len(cur), cur):
            root_to_rep_name[r] = d.original

    # For each UF group pick the minimum-size core among its members
    # as the group's signature. partition_key is taken from that member.
    root_to_sig: Dict[str, Tuple[frozenset, Tuple]] = {}
    for r, pairs in root_to_pairs.items():
        if not pairs:
            continue
        # Smallest core wins (fewest tokens, then lex for determinism)
        best = min(pairs, key=lambda cp: (len(cp[0]), tuple(sorted(cp[0]))))
        root_to_sig[r] = best

    # Bucket UF roots by signature
    buckets: Dict[Tuple, List[str]] = defaultdict(list)
    for r, sig in root_to_sig.items():
        buckets[sig].append(r)

    unions_performed = 0
    clusters_merged = 0
    log: List[Tuple] = []
    for sig, roots in buckets.items():
        if len(roots) < 2:
            continue
        # Stable pivot: root with alphabetically-smallest representative
        pivot = min(roots, key=lambda r: root_to_rep_name[r])
        names_sorted = sorted(root_to_rep_name[r] for r in roots)
        merged_here = 0
        for other in roots:
            if other == pivot:
                continue
            if uf.union(pivot, other):
                unions_performed += 1
                merged_here += 1
        if merged_here > 0:
            clusters_merged += 1
            log.append((tuple(sorted(sig[0])), names_sorted))

    return {
        "unions_performed": unions_performed,
        "clusters_merged": clusters_merged,
        "log": log,
    }


# ═══════════════════════════════════════════════════════════════════
#  Canonical selection
# ═══════════════════════════════════════════════════════════════════

def select_canonical(
    members: List[str], pmid_count: Dict[str, int],
) -> str:
    """Pick canonical by the 4-tier rule:
       1. highest PMID coverage (pmid_count[member] descending)
       2. fewest root tokens in the original string (underscore count)
       3. shortest total string
       4. lexicographically smallest

    `pmid_count` maps `original` → number of distinct PMIDs in which
    that field name was seen. Missing entries default to 0.
    """
    def sort_key(m: str) -> Tuple[int, int, int, str]:
        ntok = m.count("_") + 1
        return (-pmid_count.get(m, 0), ntok, len(m), m)
    return min(members, key=sort_key)


def select_canonicals_for_groups(
    groups: Dict[str, List[str]],
    pmid_count: Dict[str, int],
) -> Dict[str, List[str]]:
    """Return {canonical: sorted_aliases} — the final alias report."""
    out: Dict[str, List[str]] = {}
    for members in groups.values():
        canon = select_canonical(members, pmid_count)
        out[canon] = sorted(members)
    return out


# ═══════════════════════════════════════════════════════════════════
#  Diagnostics helpers
# ═══════════════════════════════════════════════════════════════════

def family_size_stats(families: Iterable[Family]) -> Dict[str, int]:
    """Quick distribution summary for logging."""
    sizes = sorted(f.size for f in families)
    if not sizes:
        return {"count": 0}
    n = len(sizes)
    return {
        "count": n,
        "min": sizes[0],
        "p50": sizes[n // 2],
        "p90": sizes[min(int(n * 0.9), n - 1)],
        "p99": sizes[min(int(n * 0.99), n - 1)],
        "max": sizes[-1],
    }


if __name__ == "__main__":
    # Tiny offline smoke-test
    from rules import decompose
    sample_keys = [
        "depth", "water_depth", "sampling_depth", "depth_m",
        "organic_carbon", "total_organic_carbon",
        "organic_nitrogen", "dissolved_organic_nitrogen",
    ]
    ds = [decompose(k) for k in sample_keys]
    H = {"depth", "water", "sampling", "organic", "carbon", "nitrogen", "total", "dissolved"}
    fams = build_families(ds, H)
    for a, f in fams.items():
        print(f"[{a}] size={f.size}: {[d.original for d in f.members]}")
    print()
    print("size stats:", family_size_stats(fams.values()))

    # Simulate an LLM partition: depth says all are one group; carbon says TOC ≡ OC;
    # organic says DON ≡ ON. Union-Find then merges.
    uf = PartitionKeyUnionFind(ds)
    fake = {
        "depth": [["depth", "water_depth", "sampling_depth", "depth_m"]],
        "carbon": [["organic_carbon", "total_organic_carbon"]],
        "nitrogen": [["organic_nitrogen", "dissolved_organic_nitrogen"]],
    }
    apply_family_partitions(uf, fake)
    grps = uf.groups()
    pmid_count = {k: 10 for k in sample_keys}
    out = select_canonicals_for_groups(grps, pmid_count)
    print()
    print("Final canonical groups:")
    for c, ms in out.items():
        print(f"  {c}: {ms}")
    print(f"Refused unions (cross-partition-key): {uf.refused_count}")
