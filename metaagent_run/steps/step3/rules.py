"""Pass A — Structural decomposition rules for Step 3.

Given a normalized metadata-key string, decompose it into:
    (root, side_tags)

where `side_tags` is a dict with keys drawn from:
    stat            — statistical / aggregation descriptor   (strip, do not split)
    unit            — measurement unit                       (strip, do not split)
    aggregation     — temporal aggregation scale             (strip, do not split)
    replicate_index — numeric replicate suffix               (strip, do not split)
    isotope         — isotope target                         (elevate, force split)
    taxon           — taxonomic-group prefix                 (elevate, force split)

"Strip" side tags are removed from the root but recorded so downstream can report
"this measurement was also reported with {sd, mean, mg_l, umol_l, ...} variants".
"Elevate" side tags remain as part of the partition key — two keys with different
isotope or taxon values are hard-constrained to different canonicals.

The rule set is intentionally the INITIAL version — it is expected to be refined
iteratively after the first diagnostic run. Any pattern matched by more than a
handful of keys but not stripped will surface in `pass_a_report.json` as an
"unstripped_tail_token" candidate for inclusion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  Tag vocabularies
# ═══════════════════════════════════════════════════════════════════

# -- English stop tokens (never count toward head-word induction) --
# These slip in because Step 2 preserves author wording, so keys like
# "distance_of_sample_to_shore" tokenize to include functional connectors.
# They are excluded from root_tokens() at the statistics layer, not
# stripped from the key string itself (the original is preserved).
#
# DELIBERATE EXCLUSIONS (ambiguous with chemical element symbols):
#   "as"  — arsenic (As); common in water contamination / sediment studies
#   "be"  — beryllium (Be)
#   "at"  — unit "atmosphere" in some pressure contexts
# We accept the false-positive cost of "as X" English phrasing becoming a
# low-DF token, in exchange for not suppressing the As concept.
STOPWORDS: frozenset = frozenset({
    "of", "in", "the", "and", "or", "to", "from", "by", "for",
    "with", "on", "a", "an", "per", "into", "onto", "up", "down",
    "is", "are", "was", "were", "not", "no",
})

# -- stat / aggregation descriptors (SUFFIX strip) ------------------
# NOTE: "n" intentionally excluded — in aquatic-chemistry convention,
# suffix "_n" usually denotes "measured as nitrogen" (e.g. no3_n),
# NOT sample count.
STAT_SUFFIXES: Tuple[str, ...] = (
    "sd", "stdev", "std", "standard_deviation",
    "se", "stderr", "standard_error",
    "err", "error",
    "mean", "avg", "average",
    "median",
    "min", "minimum",
    "max", "maximum",
    "range",
    "var", "variance",
    "cv", "rsd",
    "iqr",
    "count",
)

# -- stat descriptors (PREFIX strip) --------------------------------
# Forms like "mean_annual_temperature", "average_water_depth".
# "total_" is intentionally EXCLUDED — per dedup qualifier class (3)
# QUANTITATIVE-SUB-POOL, "total" denotes a distinct measurement
# (e.g. total_phosphorus ≠ phosphorus ≠ dissolved_phosphorus).
STAT_PREFIXES: Tuple[str, ...] = (
    "mean", "avg", "average",
    "median",
    "min", "minimum",
    "max", "maximum",
    "sum",
)

# -- measurement units (suffix-matched, strip) ---------------------
# Multi-char, unambiguous units only. Single-letter unit suffixes
# (_m, _l, _g, _s) are intentionally excluded to avoid stripping
# meaningful trailing tokens. They can be added in a later iteration
# if the report surfaces them as high-DF unstripped tails.
UNIT_SUFFIXES: Tuple[str, ...] = (
    # concentration (mass per volume)
    "mg_l", "mgl", "mgl1",
    "ug_l", "ugl", "ugl1",
    "ng_l", "ngl",
    "mg_ml",
    "g_l", "gl", "gl1",
    # concentration (molar)
    "umol_l", "umoll", "umol_per_l", "micromolar", "umolar",
    "nmol_l", "nmoll",
    "mmol_l", "mmoll",
    "mol_l", "moll",
    "pmol_l", "pmoll",
    "umol_kg", "umol_kg1", "umolkg",
    "nmol_kg", "nmolkg",
    "mmol_kg", "mmol_kg1", "mmolkg",
    "mol_kg", "mol_kg1", "molkg",
    # concentration (per mass / per volume)
    "mg_kg", "mgkg", "mgkg1",
    "ug_kg", "ugkg",
    "ng_kg", "ngkg",
    "mg_g", "mgg",
    "ug_g", "ugg",
    "mg_m3", "mgm3",
    "ug_m3", "ugm3",
    # parts-per
    "ppm", "ppb", "ppt",
    # salinity
    "psu", "practical_salinity_units",
    # length
    "m", "km", "cm", "mm", "um", "nm",
    # area
    "ha", "km2", "m2", "cm2", "mm2", "acres", "hectare", "hectares",
    # volume
    "ml", "ul", "m3", "cm3", "l",
    # mass
    "kg", "mg", "ug", "ng",
    # time / age
    "y_bp", "yr_bp", "yrs_bp", "ybp", "yrbp",
    "ka", "ma", "ky", "kyr", "kyr_bp", "kybp",
    "cal_bp", "cal_yr_bp", "calbp",
    "days", "hours", "minutes", "seconds",
    # radioactivity
    "bq_kg", "bq_l", "bq_m3", "bqkg", "bql",
    "pci_l", "pcil",
    # pressure
    "atm", "bar", "mbar", "psi", "kpa", "mpa", "pa", "hpa",
    # temperature
    "degc", "degree_celsius", "degrees_celsius", "celsius",
    "degf", "degree_fahrenheit",
    "kelvin",
    # CO2 / pCO2
    "uatm", "ppmv",
    # alkalinity
    "meq_l", "meql", "meql1",
    "mg_l_caco3", "ppm_caco3",
    # other
    "percent", "percentage", "pct",
    "ratio",
)

# -- measurement-qualifier suffixes (strip as side_tag 'qualifier') -
# These are descriptive tokens that, when appended to a substance name,
# are semantically redundant in field-discovery terms:
#   nitrate_concentration  ≡  nitrate
#   dissolved_oxygen_concentration  ≡  dissolved_oxygen
# We deliberately DO NOT include "_content" here — "water_content",
# "organic_matter_content" etc. are legitimately distinct from the
# underlying substance and should be preserved for LLM family judgement.
QUALIFIER_SUFFIXES: Tuple[str, ...] = (
    "concentration", "concentrations",
    "conc",
    "level", "levels",
    "amount", "amounts",
)

# -- temporal aggregation prefixes (prefix-matched, strip) ---------
AGGREGATION_PREFIXES: Tuple[str, ...] = (
    "annual",
    "yearly",
    "monthly",
    "weekly",
    "daily",
    "hourly",
    "seasonal",
    "diurnal",
    "mean_annual",
    "mean_monthly",
    "mean_daily",
    "average_annual",
    "average_monthly",
)

# -- isotope targets (prefix-matched, ELEVATE to split dimension) --
# A token that IS one of these, at key start (optionally followed by _),
# is extracted as isotope tag and removed from the root.
ISOTOPE_TOKENS: Tuple[str, ...] = (
    # carbon
    "13c", "14c", "12c",
    # nitrogen
    "15n", "14n",
    # oxygen
    "18o", "17o", "16o",
    # hydrogen
    "2h", "3h", "d",   # d = deuterium — risky as 1-letter, see note below
    # sulfur
    "34s", "32s", "33s", "36s",
    # cesium
    "137cs", "134cs",
    # strontium
    "87sr", "86sr",
    # lead
    "210pb", "206pb", "207pb", "208pb", "204pb",
    # radium
    "226ra", "228ra", "224ra", "223ra",
    # other common
    "40ar", "36cl", "3he", "4he", "222rn",
    "7be", "10be", "230th", "234u", "238u",
)
# NOTE: "d" alone (deuterium) is EXCLUDED from ISOTOPE_TOKENS because it
# collides with too many innocent tokens (d as in "day", "depth"...).
# Delta-notation (e.g., d13c, d15n) can be handled in a later iteration.
ISOTOPE_TOKENS = tuple(t for t in ISOTOPE_TOKENS if t != "d")

# -- taxonomic-group prefixes (prefix-matched, SOFT side-tag) ------
# NOTE: taxonomic-prefixed fields (e.g. archaeal_*, bacterial_*) describe
# community-composition measurements, which are NOT target sample
# metadata in this study. We still strip the prefix to record it as a
# side tag, but do NOT treat taxon as a hard partition-key dimension.
TAXON_PREFIXES: Tuple[str, ...] = (
    "archaeal", "archaea",
    "bacterial", "bacteria",
    "fungal", "fungi",
    "viral", "virus", "viruses",
    "eukaryotic", "eukaryote", "eukaryotes",
    "prokaryotic", "prokaryote", "prokaryotes",
    "cyanobacterial", "cyanobacteria",
    "protist", "protists",
    "microbial",
)


# -- substance tokens (extracted but NOT stripped) -----------------
# We look for these tokens inside the root and record the implied
# substance(s) as a side_tag. Substance is used as a hard partition-key
# dimension so that cross-substance merges (e.g. organic_carbon vs
# organic_nitrogen under the shared "organic" family) are refused in
# Pass B regardless of any LLM decision. Unlike stat/unit/qualifier,
# these tokens REMAIN in the root — they are concept-bearing in most
# contexts and must still participate in family membership.
#
# The map below normalizes abbreviations/ions/composite indicators to
# their principal element. A root containing no substance token
# receives substance = None (freely mergeable on that dimension).
ELEMENT_NORMALIZATION: Dict[str, str] = {
    # full element names
    "carbon": "carbon", "nitrogen": "nitrogen", "phosphorus": "phosphorus",
    "phosphorous": "phosphorus",  # common misspelling
    "sulfur": "sulfur", "sulphur": "sulfur",
    "oxygen": "oxygen", "hydrogen": "hydrogen",
    "silicon": "silicon", "silica": "silicon",
    "iron": "iron", "manganese": "manganese",
    "calcium": "calcium", "magnesium": "magnesium",
    "sodium": "sodium", "potassium": "potassium",
    "copper": "copper", "zinc": "zinc",
    "lead": "lead", "mercury": "mercury",
    "cadmium": "cadmium", "arsenic": "arsenic",
    "chromium": "chromium", "nickel": "nickel",
    "aluminum": "aluminum", "aluminium": "aluminum",
    "chlorine": "chlorine", "chloride": "chlorine",
    "fluorine": "fluorine", "fluoride": "fluorine",
    "bromine": "bromine", "bromide": "bromine",
    "boron": "boron",
    # element symbols
    "c": "carbon", "n": "nitrogen", "p": "phosphorus", "s": "sulfur",
    "o": "oxygen", "h": "hydrogen", "k": "potassium",
    "fe": "iron", "mn": "manganese", "ca": "calcium", "mg": "magnesium",
    "na": "sodium", "cu": "copper", "zn": "zinc", "pb": "lead",
    "hg": "mercury", "cd": "cadmium", "as": "arsenic",
    "cr": "chromium", "ni": "nickel",
    "al": "aluminum", "cl": "chlorine", "si": "silicon",
    "f": "fluorine", "br": "bromine", "b": "boron",
    # ions and simple molecules — normalized to their principal element
    "no3": "nitrogen", "no2": "nitrogen",
    "nh4": "nitrogen", "nh3": "nitrogen",
    "n2": "nitrogen", "n2o": "nitrogen",
    "po4": "phosphorus", "po43": "phosphorus",
    "so4": "sulfur", "so42": "sulfur", "h2s": "sulfur",
    "co2": "carbon", "co3": "carbon", "hco3": "carbon",
    "co": "carbon", "ch4": "carbon",
    "o2": "oxygen", "o3": "oxygen",
    "h2": "hydrogen",
    "sio2": "silicon", "sio3": "silicon", "sio4": "silicon",
    # compound substances
    "ammonia": "nitrogen", "ammonium": "nitrogen",
    "nitrate": "nitrogen", "nitrite": "nitrogen",
    "phosphate": "phosphorus",
    "sulfate": "sulfur", "sulphate": "sulfur",
    "carbonate": "carbon", "bicarbonate": "carbon",
    "methane": "carbon",
    # composite bulk indicators (standard env-chem abbreviations)
    "toc": "carbon", "doc": "carbon", "dic": "carbon", "poc": "carbon",
    "ton": "nitrogen", "don": "nitrogen", "tn": "nitrogen",
    "pon": "nitrogen", "tdn": "nitrogen",
    "tp": "phosphorus", "dop": "phosphorus", "tdp": "phosphorus",
    "pop": "phosphorus",
    # non-elemental but distinct substances
    "chlorophyll": "chlorophyll", "chl": "chlorophyll",
    "chla": "chlorophyll",
}
SUBSTANCE_TOKENS: frozenset = frozenset(ELEMENT_NORMALIZATION.keys())


# ═══════════════════════════════════════════════════════════════════
#  Compiled regex
# ═══════════════════════════════════════════════════════════════════

def _alt(tokens: Tuple[str, ...]) -> str:
    # Sort by descending length so longer alternatives are tried first
    # (critical for e.g. matching "umol_l" before "l").
    ordered = sorted(set(tokens), key=lambda t: (-len(t), t))
    return "|".join(re.escape(t) for t in ordered)


_STAT_SUFFIX_RE = re.compile(rf"_({_alt(STAT_SUFFIXES)})$")
_STAT_PREFIX_RE = re.compile(rf"^({_alt(STAT_PREFIXES)})_")
_UNIT_SUFFIX_RE = re.compile(rf"_({_alt(UNIT_SUFFIXES)})$")
_QUALIFIER_SUFFIX_RE = re.compile(rf"_({_alt(QUALIFIER_SUFFIXES)})$")
_AGG_PREFIX_RE = re.compile(rf"^({_alt(AGGREGATION_PREFIXES)})_")
_REPLICATE_SUFFIX_RE = re.compile(r"_(\d{1,2})$")
_ISOTOPE_PREFIX_RE = re.compile(rf"^({_alt(ISOTOPE_TOKENS)})(?:_|$)")
_TAXON_PREFIX_RE = re.compile(rf"^({_alt(TAXON_PREFIXES)})_")


# ═══════════════════════════════════════════════════════════════════
#  Decomposition
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Decomposed:
    """Structural decomposition of a normalized metadata key.

    The partition_key determines hard-constrained canonical splits
    (different partition_key → different canonical, regardless of any
    LLM decision). It includes only `isotope` and `substance`, which
    are chemistry-grounded identity dimensions where cross-value merges
    are always semantically wrong. `taxon` is a SOFT side-tag, recorded
    for completeness but not used as a split constraint. `stat`, `unit`,
    `qualifier`, `aggregation`, `replicate_index` are strip tags — they
    are removed from the root but retained for downstream reporting.
    """
    original: str
    root: str
    stat: Optional[str] = None
    unit: Optional[str] = None
    qualifier: Optional[str] = None
    aggregation: Optional[str] = None
    replicate_index: Optional[str] = None
    isotope: Optional[str] = None
    taxon: Optional[str] = None
    substance: Optional[frozenset] = None  # frozenset[str] of normalized elements

    @property
    def partition_key(self) -> Tuple[Optional[str], Optional[frozenset]]:
        """Hard-constraint dimensions: (isotope, substance).

        Two decomposed entries with different partition_keys are
        hard-constrained to different canonicals regardless of LLM
        decision. Root differences are NOT hard constraints — they are
        what the LLM's qualifier rules judge.
        """
        return (self.isotope, self.substance)

    @property
    def side_tags(self) -> Dict[str, object]:
        return {
            "stat": self.stat,
            "unit": self.unit,
            "qualifier": self.qualifier,
            "aggregation": self.aggregation,
            "replicate_index": self.replicate_index,
            "isotope": self.isotope,
            "taxon": self.taxon,
            "substance": self.substance,
        }

    def has_side_tag(self) -> bool:
        return any(v is not None for v in self.side_tags.values())


def decompose(key: str) -> Decomposed:
    """Apply the rule set to decompose `key` into (root, side_tags).

    Suffix strip order (applied once each):
        1. replicate-index numeric suffix                (area_1 → area)
        2. stat suffix                                   (X_sd   → X)
        3. qualifier suffix                              (X_concentration → X)
        4. unit suffix (iterated up to 2×)               (X_mg_l → X)

    Prefix strip loop (iterated to fixed point — max 4 passes):
        5. stat prefix                                   (mean_X → X)
        6. aggregation prefix                            (annual_X → X)
        7. isotope prefix → elevate as partition key     (13c_X → X, isotope=13c)
        8. taxonomic prefix → elevate as partition key   (archaeal_X → X, taxon=archaeal)

    Prefixes are iterated because natural compounds nest them, e.g.
    "mean_annual_X" requires strip mean → strip annual → X, and
    "annual_mean_X" requires the reverse order. Iterating to fixed point
    handles both.

    After each strip, if the remaining root would be empty or <2 chars,
    the strip is reverted.
    """
    if not key:
        return Decomposed(original=key, root=key)

    root = key
    tags: Dict[str, Optional[str]] = {
        "stat": None, "unit": None, "qualifier": None,
        "aggregation": None, "replicate_index": None,
        "isotope": None, "taxon": None,
    }

    def _try_strip(pattern: re.Pattern, group: int, tag_name: str) -> bool:
        nonlocal root
        m = pattern.search(root)
        if m is None:
            return False
        new_root = (root[: m.start()] + root[m.end():]).strip("_")
        if len(new_root) < 2:
            return False
        tags[tag_name] = m.group(group)
        root = new_root
        return True

    # Suffix pass
    _try_strip(_REPLICATE_SUFFIX_RE, 1, "replicate_index")
    _try_strip(_STAT_SUFFIX_RE, 1, "stat")
    _try_strip(_QUALIFIER_SUFFIX_RE, 1, "qualifier")
    for _ in range(2):
        before = root
        _try_strip(_UNIT_SUFFIX_RE, 1, "unit")
        if root == before:
            break

    # Prefix pass — iterate to fixed point (compositions of agg/stat prefixes)
    for _ in range(4):
        before = root
        if tags["stat"] is None:
            _try_strip(_STAT_PREFIX_RE, 1, "stat")
        if tags["aggregation"] is None:
            _try_strip(_AGG_PREFIX_RE, 1, "aggregation")
        if tags["isotope"] is None:
            _try_strip(_ISOTOPE_PREFIX_RE, 1, "isotope")
        if tags["taxon"] is None:
            _try_strip(_TAXON_PREFIX_RE, 1, "taxon")
        if root == before:
            break

    # Substance extraction — scans the root tokens and normalizes any
    # matching chemistry/ion/compound tokens to their principal element.
    # Tokens ARE kept in the root (they're concept-bearing for family
    # membership); the substance is only used as a hard partition-key
    # dimension to prevent cross-substance merges.
    substances = set()
    for tok in root.split("_"):
        if tok and tok in SUBSTANCE_TOKENS:
            substances.add(ELEMENT_NORMALIZATION[tok])
    substance_fs: Optional[frozenset] = frozenset(substances) if substances else None

    return Decomposed(
        original=key,
        root=root,
        stat=tags["stat"],
        unit=tags["unit"],
        qualifier=tags["qualifier"],
        aggregation=tags["aggregation"],
        replicate_index=tags["replicate_index"],
        isotope=tags["isotope"],
        taxon=tags["taxon"],
        substance=substance_fs,
    )


# ═══════════════════════════════════════════════════════════════════
#  Batch helper
# ═══════════════════════════════════════════════════════════════════

def decompose_all(keys: List[str]) -> List[Decomposed]:
    return [decompose(k) for k in keys]


def rule_hit_summary(decomposed: List[Decomposed]) -> Dict[str, Dict[str, int]]:
    """Count how many times each concrete tag value was assigned."""
    from collections import Counter
    summary: Dict[str, Counter] = {
        "stat": Counter(), "unit": Counter(), "qualifier": Counter(),
        "aggregation": Counter(), "replicate_index": Counter(),
        "isotope": Counter(), "taxon": Counter(), "substance": Counter(),
    }
    for d in decomposed:
        for name, value in d.side_tags.items():
            if value is None:
                continue
            if name == "substance":
                # substance is a frozenset — count element occurrences, not sets
                for elem in value:
                    summary[name][elem] += 1
            else:
                summary[name][value] += 1
    return {k: dict(v) for k, v in summary.items()}


# ═══════════════════════════════════════════════════════════════════
#  Introspection — helpful for diagnose.py
# ═══════════════════════════════════════════════════════════════════

def root_tokens(decomposed: Decomposed) -> List[str]:
    """Split the root on underscores; drop empty fragments and stopwords.

    Stopword filtering prevents English connectors (of / in / and / ...)
    from contaminating the head-word set: they would otherwise become
    the highest-DF tokens purely because authors write field names as
    natural-language phrases.
    """
    return [t for t in decomposed.root.split("_") if t and t not in STOPWORDS]


def is_orphan_root(decomposed: Decomposed, headwords: set) -> bool:
    """A root is an orphan if no token of its root is a head word."""
    return not any(t in headwords for t in root_tokens(decomposed))


if __name__ == "__main__":
    # Smoke tests
    examples = [
        "depth",
        "water_depth",
        "sampling_depth",
        "depth_below_sea_floor",
        "in_situ_temperature",
        "137cs_activity_concentration",
        "137cs_sd",
        "137cs_mean",
        "archaeal_16s_rrna_gene_copies_l",
        "bacterial_16s_rrna_gene_copies_l",
        "annual_mean_sea_surface_temperature",
        "mean_annual_sea_surface_temperature",
        "average_annual_air_temperature",
        "area_mm2",
        "area_km2",
        "area_ha",
        "ammonium_umol_per_l",
        "nitrate_umol_kg",
        "nitrate_concentration",
        "bacterial_concentration",
        "chlorophyll_a",
        "14c_age_yr_bp",
        "area_1",
        "area_2",
        "ph",
        "100_m",
        "2_m",
        "depth_m",
        "water_depth_m",
        "distance_of_sample_to_shore",   # stopword test
        "no3_n",                         # chemistry convention, _n must NOT be stripped
        "total_phosphorus",              # total_ must NOT be stripped (SPLIT qualifier)
        "total_dissolved_phosphorus",
    ]
    # Add substance-guard test cases
    examples += [
        "organic_carbon",
        "organic_nitrogen",
        "total_organic_carbon",
        "dissolved_organic_carbon",
        "dissolved_organic_nitrogen",
        "c_n_ratio",
        "nh4",
        "ammonium",
        "chlorophyll_a_concentration",
    ]
    for key in examples:
        d = decompose(key)
        sub_display = (
            "{" + ",".join(sorted(d.substance)) + "}" if d.substance else "—"
        )
        tags_list = []
        for k, v in d.side_tags.items():
            if v is None:
                continue
            if k == "substance":
                continue  # printed separately
            tags_list.append(f"{k}={v}")
        tags_str = ", ".join(tags_list) or "—"
        tokens = root_tokens(d)
        print(
            f"  {key:40s} → root={d.root!r:30s} subst={sub_display:22s} "
            f"pk={d.partition_key}  [{tags_str}]"
        )
