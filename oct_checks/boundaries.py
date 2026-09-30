"""Per-column summaries of a layer mask, and the layer-order (anatomy) check.

In plain words: In every image column: where does each layer boundary sit, and are the layers
stacked in the right order (vitreous, then the retinal bands, then the RPE)? A plausible mask never
puts a lower layer above a higher one.

Labels: 0 vitreous, 1 ILM-OPL, 2 OPL-IS/OS, 3 IS/OS-IBRPE (EZ band), 4 IBRPE-OBRPE (RPE), 5 below.
Boundary k (1 ILM, 2 OPL, 3 IS-OS, 4 IBRPE, 5 OBRPE) in a column is the number of pixels whose
label is < k: the cumulative-thickness form, ordered by construction (He et al., MICCAI 2019 and
MedIA 2021 build layer order this way). On an ordered column that is the first row whose label is
>= k, so on OCT5k's human masks it returns the traced boundary exactly. On a model mask a stray
pixel moves a boundary by only one pixel, where "first row >= k" (the pre-registered rule, kept as
method="first_row") would jump to the stray pixel's row: see analysis/prereg/phase_a_amendment1.md.
"""
import numpy as np  # arrays and maths

NAMES = ("ILM", "OPL", "IS-OS", "IBRPE", "OBRPE")  # the 5 boundaries, top to bottom
EZ, RETINA = 3, (1, 2, 3, 4)                         # the EZ band's label; the labels that are retina


def boundaries(mask: np.ndarray, method: str = "count") -> np.ndarray:
    """(H, W) labels -> (5, W) boundary rows as floats, NaN where a column has no such layer."""
    m = np.asarray(mask)                                   # accept any array-like mask
    out = np.full((len(NAMES), m.shape[1]), np.nan)        # 5 rows x W columns, "unknown" to start
    for k in range(1, len(NAMES) + 1):                     # boundary 1 (ILM) .. 5 (OBRPE)
        hit = m >= k                                       # pixels at or below this boundary
        found = hit.any(0)                                 # columns where the boundary exists at all
        # "count": number of pixels above the boundary; "first_row": the first pixel at or below it
        rows = (m < k).sum(0) if method == "count" else hit.argmax(0)
        out[k - 1, found] = rows[found]                    # store only where the boundary exists
    return out


def order_violation_columns(mask: np.ndarray) -> int:
    """The anatomy check: in how many columns does, reading top to bottom, a label follow a larger
    one (e.g. vitreous below the RPE)? Retinal layers keep their order (He et al. MICCAI 2019,
    MedIA 2021; Garvin et al. TMI 2009), and none of OCT5k's 5,016 human masks breaks it.
    Zero tolerance, because anatomy has none. Returned as a signal, not a gate criterion: on the
    model's masks, stray pixels make 54% of B-scans fail it (Phase A, A1)."""
    return int(order_violation_by_column(mask).sum())            # how many columns break the order


def order_violation_by_column(mask: np.ndarray) -> np.ndarray:
    """Per image column: does it break the anatomical layer order? (one bool per column)
    The triage gear uses it to say WHERE the order breaks, not only how often."""
    steps = np.diff(np.asarray(mask).astype(np.int16), axis=0)  # label change from each row to the next
    return (steps < 0).any(0)                                    # True where some step goes back up the order


def out_of_order_pixels(mask: np.ndarray) -> np.ndarray:
    """Per image column: how many pixels sit below a larger label (tissue out of layer order, e.g.
    vitreous under the RPE). Times the axial pixel size, it is how THICK the break is, so a stray
    pixel counts 1 and a misplaced band counts its full height."""
    m = np.asarray(mask).astype(np.int16)
    above = np.maximum.accumulate(m, axis=0)                     # the largest label at or above each pixel
    return (m < above).sum(0)                                    # pixels smaller than something above them


def column_summary(mask: np.ndarray, method: str = "count") -> dict:
    """Everything the volume checks need from one B-scan: the boundary rows, and per column whether
    it shows retina at all and whether it shows the EZ band."""
    m = np.asarray(mask)                              # accept any array-like mask
    return {"z": boundaries(m, method),               # (5, W) boundary rows
            "retina": np.isin(m, RETINA).any(0),      # per column: any retinal band?
            "ez": (m == EZ).any(0)}                   # per column: any EZ band? (missing EZ = EZ loss)
