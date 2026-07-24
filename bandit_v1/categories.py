"""Category-name canonicalization for bandit v1 (task 3 fix).

Single source of truth used by BOTH `pool.py` (pool table's `category` column) and
`states.py` (fingerprint comparison + `start_features`), per the task-3-report.md
diagnosis: 17/1516 robocasa mjcf object instances are registered under two
overlapping categories (see config.CATEGORY_ALIASES for the exact pairs and how
they were derived). Forward sampling (env.reset()'s object placement) and
env.reset_to()'s reverse mjcf_path->category lookup can legitimately disagree on
which of the two names to report for the SAME physical instance -- the mjcf path
is the identity ground truth, category is just a label on top of it, so every
consumer of `category` must canonicalize through the same alias table or the two
names will silently fail to compare/join as equal.
"""
from . import config


def canonical_category(name: str) -> str:
    """Resolve `name` to its canonical category label via config.CATEGORY_ALIASES.
    Categories with no alias (the overwhelming majority) pass through unchanged."""
    return config.CATEGORY_ALIASES.get(name, name)
