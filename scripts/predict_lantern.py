"""Produce point predictions and calibrated intervals for new participants."""

import argparse
import pickle
import numpy as np
import pandas as pd
from lantern.fir import estimate_fir_area, response_error_budget
from lantern.reserve import build_reserve_features
from lantern.calibration import conformal_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--bold-dir", required=True)
    parser.add_argument("--design-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    decoder = model["decoder"]
    cfg = model["config"]

    def reserve_matrix(subject_id):
        ts = np.load(f"{args.bold_dir}/{subject_id}_ts.npy")
        design = np.load(f"{args.design_dir}/{subject_id}_design.npy")
        M_hat, eps, _ = response_error_budget(
            ts, design, cfg["fir"]["tr"], cfg["fir"]["horizon"],
            n_bootstrap=cfg["reserve"]["n_bootstrap"]
        )
        intervals, _ = build_reserve_features(
            M_hat, eps, cfg["anatomy"]["groups"], cfg["reserve"]["tau"]
        )
        lo = np.asarray([intervals[k][0] for k in model["reserve_keys"]])
        hi = np.asarray([intervals[k][1] for k in model["reserve_keys"]])
        return lo, hi

    def load_df(path):
        df = pd.read_csv(path, sep="\t")
        z = df[cfg["outcome"]["clinical_context"]].values.astype(float)
        y = df[cfg["outcome"]["target"]].values.astype(float)
        return df, z, y

    df_cal, z_cal, y_cal = load_df(args.calib)
    lo_cal = np.stack([reserve_matrix(sid)[0] for sid in df_cal["subject_id"]])
    hi_cal = np.stack([reserve_matrix(sid)[1] for sid in df_cal["subject_id"]])

    interval_fn, q = conformal_interval(
        decoder,
        z_cal,
        lo_cal,
        hi_cal,
        y_cal,
        alpha=cfg["calibration"]["alpha"],
        clip=tuple(cfg["calibration"]["clip"]),
    )

    df_test, z_test, y_test = load_df(args.test)
    lo_test = np.stack([reserve_matrix(sid)[0] for sid in df_test["subject_id"]])
    hi_test = np.stack([reserve_matrix(sid)[1] for sid in df_test["subject_id"]])

    point = 0.5 * (decoder.predict(z_test, lo_test) + decoder.predict(z_test, hi_test))
    lower, upper = interval_fn(z_test, lo_test, hi_test)

    out = df_test[["subject_id"]].copy()
    out["point"] = point
    out["lower"] = lower
    out["upper"] = upper
    out["true"] = y_test
    out.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    main()
