from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityConfig:
    # Quality thresholds
    quality_min: float = 0.5

    # Aggregation thresholds
    outlier_threshold: float = 0.35  # minimum cosine similarity to the median embedding
    n_min: int = 5
    dispersion_max: float = 0.35

    # Matching thresholds
    accept_threshold: float = 0.55
    margin_threshold: float = 0.08

    # Template maintenance
    add_threshold: float = 0.85
    max_templates_per_user: int = 12
