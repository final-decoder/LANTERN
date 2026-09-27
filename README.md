# LANTERN

Gain-invariant reserve features with uncertainty propagation for task-fMRI-based clinical prediction.

## Pipeline

1. **FIR estimation** (`lantern/fir.py`) — per-region impulse responses, integrated over a 40-s horizon; block bootstrap provides the response-error budget.
2. **Reserve construction** (`lantern/reserve.py`) — gain-invariant log-volume features as intervals, tightened by submodularity and monotonicity.
3. **Decoding** (`lantern/decoder.py`) — monotone spline decoder mapping reserve intervals plus clinical context to outcomes.
4. **Calibration** (`lantern/calibration.py`) — conformal prediction intervals on held-out participants.

