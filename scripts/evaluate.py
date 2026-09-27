"""Evaluate point predictions, intervals, and uncertainty retention."""

import argparse
import numpy as np
import pandas as pd
from lantern.utils import mae_r2, coverage_width, retention_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", default="true")
    args = parser.parse_args()

    df = pd.read_csv(args.pred, sep="\t")
    y = df[args.target].values
    point = df["point"].values
    lower = df["lower"].values
    upper = df["upper"].values

    mae, r2 = mae_r2(y, point)
    cov, width = coverage_width(lower, upper, y)
    uncertainty = upper - lower
    errors = np.abs(y - point)
    frac, retained = retention_curve(errors, uncertainty)

    print(f"MAE: {mae:.3f}")
    print(f"R2 : {r2:.3f}")
    print(f"Coverage: {cov:.3f}")
    print(f"Mean width: {width:.3f}")
    print("Retention (fraction -> MAE):")
    for f, m in zip(frac, retained):
        print(f"  {f:.2f} -> {m:.3f}")


if __name__ == "__main__":
    main()
