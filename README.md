# LANTERN

Gain-invariant reserve features with uncertainty propagation for task-fMRI-based clinical prediction.

## Pipeline

1. **FIR estimation** (`lantern/fir.py`) — per-region impulse responses, integrated over a 40-s horizon; block bootstrap provides the response-error budget. Cosine drift removal and frame scrubbing are built in.
2. **Reserve construction** (`lantern/reserve.py`) — gain-invariant log-volume features as intervals, tightened by submodularity and monotonicity; includes tau-sensitivity scans and leave-one-group-out diagnostics.
3. **Decoding** (`lantern/decoder.py`) — monotone spline decoder mapping reserve intervals plus clinical context to outcomes, with cross-validated hyperparameter selection.
4. **Calibration** (`lantern/calibration.py`) — split-conformal or CV+ prediction intervals on held-out participants.

Image I/O (`lantern/io.py`) extracts parcel time series from NIfTI volumes and builds condition designs from BIDS events files.

## Install

```bash
pip install -e .
```

## Quick start

```bash
python examples/demo_synthetic.py
python examples/demo_sensitivity.py
```

## Tests

```bash
pytest
```

Full runs are driven by `scripts/fit_lantern.py`, `scripts/predict_lantern.py`, and `scripts/evaluate.py` with `config/default.yaml`.
