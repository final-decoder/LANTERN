"""Fit LANTERN reserve and decoder on a training cohort."""

import argparse
import pickle
import yaml
import numpy as np
from lantern.fir import estimate_fir_area, response_error_budget
from lantern.reserve import build_reserve_features
from lantern.decoder import MonotoneSplineDecoder
from lantern.utils import group_unions


def load_cohort(tsv_path, context_cols, target_col):
    """Load tabular clinical data; returns (ids, z, y)."""
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t")
    z = df[context_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    return df, z, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bold-dir", required=True)
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df, z, y = load_cohort(
        args.clinical,
        cfg["outcome"]["clinical_context"],
        cfg["outcome"]["target"],
    )

    # Build per-subject reserve features: parcel-level time series are
    # lesion-masked, then n_regions x n_conditions reserve intervals are
    # computed for every nonempty group union.
    reserve_lo = []
    reserve_hi = []
    for subject_id in df["subject_id"]:
        # Load pre-extracted per-parcel time series and condition design.
        ts = np.load(f"{args.bold_dir}/{subject_id}_ts.npy")
        design = np.load(f"{args.design_dir}/{subject_id}_design.npy")
        M_hat, eps, _ = response_error_budget(
            ts,
            design,
            cfg["fir"]["tr"],
            cfg["fir"]["horizon"],
            n_bootstrap=cfg["reserve"]["n_bootstrap"],
        )
        intervals, _ = build_reserve_features(
            M_hat, eps, cfg["anatomy"]["groups"], cfg["reserve"]["tau"]
        )
        reserve_lo.append([v[0] for v in intervals.values()])
        reserve_hi.append([v[1] for v in intervals.values()])

    lo = np.asarray(reserve_lo)
    hi = np.asarray(reserve_hi)

    decoder = MonotoneSplineDecoder(
        n_context=z.shape[1],
        n_reserve=lo.shape[1],
        knots=cfg["decoder"]["knots"],
        l2_lambda=cfg["decoder"]["l2_lambda"],
    )
    decoder.fit(
        z,
        lo,
        hi,
        y,
        max_iter=cfg["decoder"]["max_iter"],
        lr=cfg["decoder"]["learning_rate"],
    )

    model = {
        "decoder": decoder,
        "config": cfg,
        "reserve_keys": list(group_unions(cfg["anatomy"]["groups"]).keys()),
    }
    with open(args.out, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
