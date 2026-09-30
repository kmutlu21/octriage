"""Neighbour check (A2b): is a B-scan continuous with the B-scans next to it? The limits are learned
from the graders' own tracings, never chosen by us.

In plain words: Neighbouring B-scans are slices of the same eye a fraction of a millimetre apart, so
a boundary should not jump far from one to the next. This check learns how far the graders' own
tracings move between neighbours and warns when a model's mask moves further.

Garvin et al., IEEE TMI 2009 (doi:10.1109/TMI.2009.2016958), section III-C: feasibility limits are
learned from reference tracings as mean ± 2.6 SD, "to allow for 99% of the expected" values. Here,
per boundary: how much its depth changes from one B-scan to the next, at the same column. One
constant limit over all columns (Garvin's original form): volumes of 19 to 61 B-scans share no
common grid for his per-column version.

Then, ours but learned by the same rule: a pair of B-scans is flagged if the share of its columns
outside those limits exceeds mean + 2.6 SD of that share over the graders' own pairs (about 1% of
the graders' own columns fall outside by construction, so one column cannot be the test). A B-scan
is flagged when every neighbour pair it has is flagged: one bad B-scan breaks continuity with both
of its neighbours. Tested at matched workload in Phase A: no gain as a gate rule, so it is a signal.
"""
from dataclasses import asdict, dataclass  # a small record type, and turning it into a dict

import numpy as np  # arrays and maths

Z = 2.6  # Garvin's multiplier: mean ± 2.6 SD covers ~99% of normal values


@dataclass
class Limits:
    mu: list           # per boundary: mean depth change between neighbouring B-scans (µm)
    sd: list           # per boundary: its standard deviation (µm)
    pair_limit: float  # largest acceptable share of out-of-limit columns in one pair of B-scans
    n_pairs: int = 0   # how many grader B-scan pairs the limits were learned from

    def to_dict(self) -> dict:
        return asdict(self)  # a plain dict, so it can be saved as JSON


def deltas(z: np.ndarray, axial_um: float) -> np.ndarray:
    """z (n_bscans, 5, W) boundary rows, NaN where missing -> (n_bscans - 1, 5, W) changes in µm."""
    return (z[1:] - z[:-1]) * axial_um  # each B-scan minus the one before it, rows -> µm


def violating_fraction(d: np.ndarray, mu, sd) -> np.ndarray:
    """Per pair: the share of comparable columns where any boundary leaves mu ± 2.6 SD.
    NaN for a pair with no comparable column (a missing B-scan, or no retina in either)."""
    mu, sd = np.asarray(mu)[:, None], np.asarray(sd)[:, None]      # shape (5, 1): one value per boundary
    with np.errstate(invalid="ignore"):                             # NaN comparisons are expected; stay quiet
        outside = ((d < mu - Z * sd) | (d > mu + Z * sd)).any(1)    # (pairs, W): any boundary out of limits?
    judged = np.isfinite(d).any(1)                                  # (pairs, W): could this column be compared?
    n = judged.sum(1)                                               # comparable columns per pair
    return np.where(n > 0, (outside & judged).sum(1) / np.maximum(n, 1), np.nan)  # share, or NaN if none


def learn(volumes: list, axial_um: float) -> Limits:
    """volumes: grader boundary volumes z (n_bscans, 5, W), all at one B-scan spacing."""
    ds = [deltas(z, axial_um) for z in volumes]                    # changes between neighbours, per volume
    mu, sd = [], []                                                # one mean and SD per boundary
    for k in range(ds[0].shape[1]):                                # for each of the 5 boundaries
        v = np.concatenate([d[:, k].ravel() for d in ds])          # all its changes, all volumes
        v = v[np.isfinite(v)]                                      # drop the unknown ones
        mu.append(float(v.mean()))                                 # typical change
        sd.append(float(v.std(ddof=1)))                            # its spread
    f = np.concatenate([violating_fraction(d, mu, sd) for d in ds])  # out-of-limit share of every grader pair
    f = f[np.isfinite(f)]                                          # drop pairs that could not be judged
    return Limits(mu, sd, float(f.mean() + Z * f.std(ddof=1)), int(f.size))  # the pair limit, by the same rule


def flag_bscans(z: np.ndarray, limits: Limits, axial_um: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (per-pair out-of-limit share, per-B-scan flag). A B-scan is flagged if it has at least
    one comparable neighbour pair and every comparable neighbour pair is over the limit."""
    frac = violating_fraction(deltas(z, axial_um), limits.mu, limits.sd)  # share per neighbour pair
    bad = frac > limits.pair_limit                                        # pairs over the limit
    has = np.isfinite(frac)                                               # pairs that could be judged
    n = len(z)                                                            # number of B-scans
    flags = np.zeros(n, bool)                                             # no B-scan flagged yet
    for i in range(n):                                                    # B-scan i sits in pairs i-1 and i
        pairs = [p for p in (i - 1, i) if 0 <= p < n - 1 and has[p]]      # its judged pairs
        flags[i] = bool(pairs) and all(bad[p] for p in pairs)             # flagged only if all of them are bad
    return frac, flags
