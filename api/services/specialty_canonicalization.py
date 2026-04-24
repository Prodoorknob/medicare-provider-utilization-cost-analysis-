"""Canonicalize CMS specialty-name splits where LabelEncoder produced duplicate indices.

CMS changed the raw string for several specialties between years; the clinical specialty
did not change, but LabelEncoder gave each string its own index, producing pairs with
partial year coverage. The API treats the less-populated variant as an alias of the
more-populated canonical index and unions rows across the pair.

Proper fix (silver-layer name canonicalization + model retrain) is tracked in
MODELING.md backlog.
"""

# alias_idx -> canonical_idx
ALIAS_TO_CANONICAL: dict[int, int] = {
    17: 16,   # "Cardiovascular Disease (Cardiology)" -> "Cardiology"
    29: 28,   # "Colorectal Surgery (Proctology)"     -> "Colorectal Surgery (Formerly Proctology)"
    88: 87,   # "Oral Surgery (Dentists Only)"        -> "Oral Surgery (Dentist Only)"
}

ALIAS_IDXS: frozenset[int] = frozenset(ALIAS_TO_CANONICAL.keys())

# canonical_idx -> full list of indices to query (canonical + its aliases)
CANONICAL_TO_INDICES: dict[int, list[int]] = {}
for _alias, _canonical in ALIAS_TO_CANONICAL.items():
    CANONICAL_TO_INDICES.setdefault(_canonical, [_canonical]).append(_alias)


def canonicalize_idx(idx: int) -> int:
    """Map an alias idx to its canonical. Non-alias idxs pass through unchanged."""
    return ALIAS_TO_CANONICAL.get(idx, idx)


def expand_canonical(idx: int) -> list[int]:
    """Return the indices that should be unioned when querying for `idx`."""
    canonical = canonicalize_idx(idx)
    return CANONICAL_TO_INDICES.get(canonical, [canonical])
