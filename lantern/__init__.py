"""Task-conditioned network reserve for language-outcome prediction."""

__version__ = "0.1.0"

from .reserve import (
    row_projector,
    reserve_log_volume,
    reserve_intervals,
    tighten_intervals,
    normalize_reserve,
)
from .decoder import MonotoneSplineDecoder, worst_case_loss
from .fir import estimate_fir_area, response_error_budget
from .calibration import conformal_interval
from .utils import group_unions, bootstrap_fir

__all__ = [
    "row_projector",
    "reserve_log_volume",
    "reserve_intervals",
    "tighten_intervals",
    "normalize_reserve",
    "MonotoneSplineDecoder",
    "worst_case_loss",
    "estimate_fir_area",
    "response_error_budget",
    "conformal_interval",
    "group_unions",
    "bootstrap_fir",
]
