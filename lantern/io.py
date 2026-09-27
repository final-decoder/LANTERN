"""Image I/O: parcel time series extraction and design loading."""

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn.signal import clean


def load_atlas_labels(atlas_img):
    """Read an integer atlas volume and return its sorted label list.

    Label 0 is treated as background and excluded.
    """
    data = np.asarray(atlas_img.dataobj).astype(int)
    labels = sorted(int(v) for v in np.unique(data) if v != 0)
    return labels


def extract_parcel_timeseries(bold_img, atlas_img, standardize=True,
                              high_pass=None, tr=None, t_r=None):
    """Extract per-parcel mean time series from a 4-D BOLD volume.

    Labels are taken from ``atlas_img``; voxels outside the atlas are
    ignored. Signals are detrended/standardized through nilearn's
    ``clean``; when ``high_pass`` is given with the repetition time,
    low-frequency drift is removed at that cutoff.

    Parameters
    ----------
    bold_img : nibabel image
        4-D BOLD series.
    atlas_img : nibabel image
        3-D integer parcellation in the same space as ``bold_img``.
    standardize : bool
    high_pass : float, optional
        Drift cutoff in Hz, passed to ``nilearn.signal.clean``.
    tr, t_r : float, optional
        Repetition time in seconds. ``t_r`` is the spelling used by
        nilearn; ``tr`` is accepted as an alias.

    Returns
    -------
    ts : (n_timepoints, n_parcels) array
        Parcel time series ordered by ascending atlas label.
    labels : list of int
        Atlas labels corresponding to the columns of ``ts``.
    """
    t_r = t_r if t_r is not None else tr
    if high_pass is not None and t_r is None:
        raise ValueError("tr is required when high_pass is set")

    bold = np.asarray(bold_img.dataobj, dtype=float)
    atlas = np.asarray(atlas_img.dataobj).astype(int)
    if bold.ndim != 4:
        raise ValueError("bold_img must be 4-D")
    if atlas.shape != bold.shape[:3]:
        raise ValueError("atlas and bold shapes do not align")

    labels = load_atlas_labels(atlas_img)
    n_time = bold.shape[3]
    ts = np.zeros((n_time, len(labels)))
    for i, lab in enumerate(labels):
        mask = atlas == lab
        ts[:, i] = bold[mask].mean(axis=0)

    ts = clean(
        ts,
        t_r=t_r,
        high_pass=high_pass,
        standardize=bool(standardize),
        detrend=True,
    )
    return ts, labels


def load_event_design(events_tsv, n_time, tr, condition_col="trial_type",
                      onset_col="onset", duration_col="duration"):
    """Boxcar-condition design matrix from a BIDS-style events file.

    Parameters
    ----------
    events_tsv : str or pandas.DataFrame
    n_time : int
        Number of BOLD frames.
    tr : float
        Repetition time in seconds.
    condition_col, onset_col, duration_col : str

    Returns
    -------
    design : (n_conditions, n_time) array
        One row per unique condition, 1 during condition blocks.
    conditions : list of str
    """
    events = (
        pd.read_csv(events_tsv, sep="\t")
        if isinstance(events_tsv, str)
        else events_tsv
    )
    conditions = sorted(events[condition_col].unique())
    design = np.zeros((len(conditions), n_time))
    t = np.arange(n_time) * tr
    for j, cond in enumerate(conditions):
        sel = events[events[condition_col] == cond]
        for _, row in sel.iterrows():
            on, dur = row[onset_col], row[duration_col]
            design[j, (t >= on) & (t < on + dur)] = 1.0
    return design, conditions
