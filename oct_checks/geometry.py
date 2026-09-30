"""OCT5k scan geometry: how big one pixel is, in micrometres (µm).

In plain words: To turn pixels into micrometres (µm) we need each pixel's size. These numbers come
from the dataset paper; on real data the gear reads them from the file's header instead.

From the OCT5k paper (Sci Data 2025, doi:10.1038/s41597-024-04259-z): Spectralis device, "axial
resolution is 3.5 µm", "scan-dimension is 8.9×7.4 mm²". Ours, and flagged as inference: 8.9 mm runs
along a B-scan and 7.4 mm across the B-scans, and 3.5 µm is taken as the pixel spacing (Spectralis
often samples ~3.9 µm), so µm and mm² values carry about ±10%. In deployment the triage gear reads
the real spacing from the volume's header instead of these constants.
"""
import numpy as np  # arrays and maths

AXIAL_UM = 3.5                 # µm per pixel, top to bottom of a B-scan (depth into the retina)
SCAN_WIDTH_UM = 8900           # the 8.9 mm one B-scan spans, left to right
LATERAL_UM = SCAN_WIDTH_UM / 512  # µm per pixel on the mask's 512 columns (~17.4 µm); source images have 496
SCAN_WIDTH_ACROSS_UM = 7400    # the 7.4 mm that the B-scans of one volume are spread across


def bscan_spacing_um(n_bscans: int) -> float:
    """Distance between two neighbouring B-scans: 7.4 mm divided into (n - 1) gaps.
    19 B-scans -> 411 µm apart; 61 B-scans -> 123 µm apart."""
    return SCAN_WIDTH_ACROSS_UM / (n_bscans - 1)  # n B-scans have n - 1 gaps between them


def en_face_coords(n_bscans: int, width: int, spacing_um: float, lateral_um: float = LATERAL_UM):
    """Position of every point of the en face (top-down) map, in µm from the scan centre:
    y for each B-scan (index 0 = the first B-scan, b001), x for each image column."""
    y = (np.arange(n_bscans) - (n_bscans - 1) / 2) * spacing_um  # B-scan number -> µm, centre = 0
    x = (np.arange(width) - (width - 1) / 2) * lateral_um        # column number -> µm, centre = 0
    return y, x                                                   # two 1-D arrays of coordinates
