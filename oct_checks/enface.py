"""Visit endpoints (A2): the en face (top-down) maps of one volume and the trial endpoints read
from them.

In plain words: Seen from above ("en face"), the volume becomes a map of retinal thickness. This
file finds the fovea (the centre of vision), reads CST (the mean thickness in the central 1-mm
circle, a trial endpoint) and the area where the EZ layer is missing (another endpoint).

- Thickness = ILM to the top of the RPE (IBRPE), the central-subfield definition of Pak, ..., Danis,
  IOVS 2013;54:4512-8 (UW Fundus Photograph Reading Center, WRC's predecessor).
- Fovea (OURS; Pak placed it by hand): the thinnest point within 1.5 mm of the scan centre.
  Known to fail in thick DME, where the thinnest point is not the pit.
- Decentred: fovea more than 200 µm from the scan centre, where Pak found the central-subfield
  thickness goes significantly wrong. Only assessable when B-scans are at most 200 µm apart: a
  smaller displacement than the sampling step cannot be measured (ours; Amendment 1).
- CST: mean thickness within 500 µm of the fovea (the central 1-mm subfield), the grid re-centred
  on the fovea as Pak recommends. Sampled points only, no interpolation between B-scans.
- EZ-loss area: en face points with retina but no EZ band, times the area of one point (the FDA-
  approved Encelto endpoint is the area of EZ loss; the team's ez-rpe-reportgear takes pixel area
  from the spacing).
"""
import numpy as np  # arrays and maths

from oct_checks.geometry import en_face_coords  # µm coordinates of every en face point

FOVEA_SEARCH_UM, CENTRAL_RADIUS_UM, DECENTRATION_UM = 1500, 500, 200  # search radius; 1-mm circle; Pak's limit


def maps(summaries: list, usable: np.ndarray, axial_um: float) -> tuple[np.ndarray, np.ndarray]:
    """summaries: per B-scan position a qc.boundaries.column_summary dict, or None if missing.
    usable: per position, may the B-scan enter the map (it passed the retina check).
    -> thickness (n, W) in µm (NaN where unknown), EZ-loss (n, W) bool."""
    width = next(s["z"].shape[1] for s in summaries if s is not None)  # columns per B-scan
    thick = np.full((len(summaries), width), np.nan)                     # thickness map, unknown to start
    ez_loss = np.zeros((len(summaries), width), bool)                    # EZ-loss map, none to start
    for i, s in enumerate(summaries):                                    # one row of the map per B-scan
        if s is None or not usable[i]:                                   # missing, or failed the retina check
            continue                                                     # leave that row unknown
        thick[i] = (s["z"][3] - s["z"][0]) * axial_um                    # IBRPE row minus ILM row, in µm
        ez_loss[i] = s["retina"] & ~s["ez"]                              # retina present but no EZ band
    return thick, ez_loss


def endpoints(thick: np.ndarray, ez_loss: np.ndarray, spacing_um: float, lateral_um: float,
              decentration_needs_sampling: bool = True) -> dict:
    """The visit's endpoints and centring. decentration_needs_sampling=False is the pre-registered
    rule (always assessed); True is Amendment 1 (only where the B-scan spacing can show 200 µm)."""
    n, width = thick.shape                                              # B-scans x columns
    assessable = not decentration_needs_sampling or spacing_um <= DECENTRATION_UM  # can we see 200 µm?
    y, x = en_face_coords(n, width, spacing_um, lateral_um)             # µm coordinates
    yy, xx = np.meshgrid(y, x, indexing="ij")                           # coordinates for every map point
    ez_area = float(ez_loss.sum() * lateral_um * spacing_um / 1e6)      # EZ-loss points x point area, in mm²
    search = (np.hypot(yy, xx) <= FOVEA_SEARCH_UM) & np.isfinite(thick) # known points near the centre
    if not search.any():  # no usable retina near the centre: the grid cannot be placed at all
        return {"fovea_found": False, "decentration_assessable": assessable, "decentred": True,
                "fovea_bscan": None, "fovea_x_um": None,
                "fovea_y_um": None, "fovea_offset_um": None, "cst_um": None, "ez_loss_mm2": ez_area,
                "central_bscans": []}
    i, j = np.unravel_index(np.argmin(np.where(search, thick, np.inf)), thick.shape)  # thinnest point = fovea
    fy, fx = float(y[i]), float(x[j])                                   # fovea position in µm
    central = np.hypot(yy - fy, xx - fx) <= CENTRAL_RADIUS_UM           # points inside the 1-mm circle
    vals = thick[central & np.isfinite(thick)]                          # known thicknesses inside it
    offset = float(np.hypot(fy, fx))                                    # fovea's distance from the scan centre
    return {"fovea_found": True, "decentration_assessable": assessable,
            "decentred": assessable and offset > DECENTRATION_UM,       # > 200 µm off centre (Pak 2013)
            "fovea_bscan": int(i) + 1,                                  # which B-scan holds the fovea (1-based)
            "fovea_x_um": fx, "fovea_y_um": fy, "fovea_offset_um": offset,
            "cst_um": float(vals.mean()) if vals.size else None,        # the CST endpoint
            "ez_loss_mm2": ez_area,                                     # the EZ-loss endpoint
            # B-scans that pass through the 1-mm circle: a flag on one of these can change the CST
            "central_bscans": [int(k) + 1 for k in np.flatnonzero(np.abs(y - fy) <= CENTRAL_RADIUS_UM)]}
