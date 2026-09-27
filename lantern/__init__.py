"""Task-conditioned network reserve for language-outcome prediction."""

__version__ = "0.2.0"

from .reserve import (
    row_projector,
    reserve_log_volume,
    reserve_intervals,
    tighten_intervals,
    normalize_reserve,
    tau_sensitivity,
    leave_one_group_out_volumes,
)
from .decoder import MonotoneSplineDecoder, worst_case_loss
from .fir import (
    estimate_fir_area,
    response_error_budget,
    make_drift_basis,
    detect_outlier_frames,
)
from .calibration import conformal_interval, cv_plus_interval
from .utils import group_unions, bootstrap_fir
from .io import extract_parcel_timeseries, load_event_design, load_atlas_labels

__all__ = [
    "row_projector",
    "reserve_log_volume",
    "reserve_intervals",
    "tighten_intervals",
    "normalize_reserve",
    "tau_sensitivity",
    "leave_one_group_out_volumes",
    "MonotoneSplineDecoder",
    "worst_case_loss",
    "estimate_fir_area",
    "response_error_budget",
    "make_drift_basis",
    "detect_outlier_frames",
    "conformal_interval",
    "cv_plus_interval",
    "group_unions",
    "bootstrap_fir",
    "extract_parcel_timeseries",
    "load_event_design",
    "load_atlas_labels",
]
