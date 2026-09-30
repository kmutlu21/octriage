"""The triage gear's module for OCT retinal layers: everything specific to OCT volumes.

In plain words: triage.py makes the decisions that apply to any imaging; this file holds everything
OCT-specific: the retina check, the layer-order and neighbour warnings, CST and EZ loss, and which
priority a visit gets.

Called by triage.py. The rules live in oct_checks/ and are the ones pre-registered in
analysis/prereg/phase_a.md (+ Amendment 1); this file only applies them to one visit and writes
each result as a checklist line that says where to look.

  gate (decides "read first")   retina found in >= min_retina_fraction of columns (a hard failure)
  signals (never change a verdict or the priority; tested in Phase A, no gain as gate rules):
      layer order per B-scan: columns where >= 20.1 µm of tissue is out of order, located along
      the B-scan (He et al. 2019: layers keep their order; 20.1 µm = CST repeatability, so stray
      pixels do not count)
      continuity with the neighbouring B-scans (Garvin et al. 2009 limits, learned from graders)
  findings (reported, never errors)
      EZ not seen over part of a B-scan: EZ loss is a disease finding and a trial endpoint
      (the FDA-approved Encelto endpoint), or a segmentation error; the grader decides
  visit: CST and EZ-loss area (oct_checks/enface.py), the reading priority (oct_checks/worklist.py), decentration
  not assessed here: image quality (no validated gradability model), change since the last visit
"""
import io  # an in-memory file, to return the PNG as bytes

import numpy as np                   # arrays and maths
from PIL import Image, ImageDraw     # drawing the en face map

from oct_checks import boundaries as qb      # boundary rows and the layer-order check
from oct_checks import continuity, enface, worklist  # neighbour check; endpoints; tiers

MISSING = 255  # label of a B-scan position that was not acquired (the relay writes it)
RAMP = ((0xcd, 0xe2, 0xfb), (0x0d, 0x36, 0x6b))  # colour ramp, one hue: light (thin) -> dark (thick)
SCALE_UM = (150.0, 550.0)  # fixed colour scale in µm, so maps of different visits compare
EZ_FINDING_MIN = 0.05  # report EZ absence over >= 5% of the retina columns (ours; the 5% of gpu_service/unet/data.py)
SEVERE_UM = 20.1  # a layer-order break counts from this thickness: CST repeatability (Domalpally et al. OSLI 2010)


def spacing_um(header: dict) -> tuple[float, float, float]:
    """(across B-scans, axial, lateral) in µm from an NRRD header written with 'spacings' in mm."""
    units = list(header.get("units", ["mm"] * 3))                  # units per axis (default: mm)
    if any(u != "mm" for u in units):                              # refuse anything we might misread
        raise ValueError(f"expected spacings in mm, got units {units}")
    across, axial, lateral = (float(s) * 1000 for s in header["spacings"])  # mm -> µm
    return across, axial, lateral


def is_missing(mask: np.ndarray) -> bool:
    """A position the relay marked as not acquired."""
    return bool((mask == MISSING).all())


def slice_gates(answer: dict, config: dict) -> list[dict]:
    """The OCT plausibility gate, on the number the model service reports. Why it exists: the
    confidence measures stability, not correctness, and a model that finds no retina is perfectly
    stable (a random-weight U-Net scored 1.0). All 5,016 OCT5k human gradings show retina in
    >= 94.9% of columns, so 0.9 only catches masks no grader would draw."""
    min_retina = float(config.get("min_retina_fraction", 0.9))
    r = answer["measurements"].get("retina_present_fraction")      # the service's plausibility number
    if r is None:
        raise ValueError("OCT triage needs retina_present_fraction from the model service")
    ok = r >= min_retina
    return [{"check": "retina", "role": "gate", "hard": True, "state": "pass" if ok else "fail",
             "retina_present_fraction": r,
             "text": f"{'✓' if ok else '✗'} retina found in {r:.0%} of columns"
                     + ("" if ok else f" (min {min_retina:.0%})")}]


def continuity_signal(summaries: list, axial_um: float, limits: dict | None) -> dict:
    """Neighbour check on the model's boundaries. limits: the JSON Phase A learns from the graders'
    tracings ({"by_n_bscans": {n: Limits}, "pooled": Limits}). Without limits it is not assessed:
    in deployment they must be learned from the site's own graders."""
    if not limits:                                                 # no limits supplied
        return {"assessed": False, "reason": "no continuity limits learned from graders at this site",
                "flagged_bscans": []}
    n = len(summaries)                                             # B-scans in the volume
    table = limits["by_n_bscans"].get(str(n), limits["pooled"])    # limits for this volume size, else pooled
    lim = continuity.Limits(**table)                               # back into the Limits record
    width = next(s["z"].shape[1] for s in summaries if s is not None)  # columns per B-scan
    z = np.stack([s["z"] if s is not None else np.full((5, width), np.nan) for s in summaries])  # (n, 5, W)
    _, flags = continuity.flag_bscans(z, lim, axial_um)            # per-B-scan flag
    return {"assessed": True, "flagged_bscans": [int(i) + 1 for i in np.flatnonzero(flags)],  # 1-based
            "limits": "Garvin 2009 rule, learned from the graders' tracings"}


def _bscans(bs: list) -> str:
    """[6] -> 'B-scan 6'; [8, 9, 10, 14] -> 'B-scans 8-10, 14'."""
    return ("B-scan " if len(bs) == 1 else "B-scans ") + _ranges(bs)


def _mm(v: float) -> str:
    """-0.03 -> '+0.0' (never '-0.0'); 1.25 -> '+1.2'."""
    return f"{round(float(v), 1) + 0.0:+.1f}"                     # + 0.0 turns -0.0 into 0.0


def _ranges(bs: list) -> str:
    """[8, 9, 10, 14] -> '8-10, 14'."""
    out, start = [], None
    for i, b in enumerate(bs):
        start = b if start is None else start
        if i + 1 == len(bs) or bs[i + 1] != b + 1:                 # the run ends here
            out.append(f"{start}" if start == b else f"{start}-{b}")
            start = None
    return ", ".join(out)


def volume_maps(labels: np.ndarray, axial: float, min_retina: float):
    """Per-B-scan summaries and the en face maps of one volume.
    Returns (summaries, missing B-scans, {B-scan: columns >= SEVERE_UM out of order}, thickness map,
    EZ-loss map)."""
    summ, usable, missing, order = [], [], [], {}                  # summaries; map entry?; missing; order columns
    for i, sl in enumerate(labels):                                # each B-scan of the volume
        if is_missing(sl):                                         # a position with no B-scan
            summ.append(None)
            usable.append(False)
            missing.append(i + 1)
            continue
        s = qb.column_summary(sl)                                  # boundaries, retina and EZ per column
        summ.append(s)
        usable.append(bool(s["retina"].mean() >= min_retina))     # the retina check decides map entry
        cols = qb.out_of_order_pixels(sl) * axial >= SEVERE_UM     # anatomy: columns broken by >= 20.1 µm
        if cols.any():
            order[i + 1] = cols
    thick, loss = enface.maps(summ, np.array(usable), axial)       # en face thickness and EZ-loss maps
    return summ, missing, order, thick, loss


def assess(labels: np.ndarray, across: float, axial: float, lateral: float, flagged: dict,
           config: dict, visit: str, limits: dict | None) -> tuple[dict, dict, list, dict]:
    """One visit: endpoints, tier, signals and findings. flagged = {B-scan: reason labels} from the
    gate. Returns (visit findings, {B-scan: extra checklist lines}, visit lines, {file name: bytes})."""
    min_retina = float(config.get("min_retina_fraction", 0.9))
    summ, missing, order, thick, loss = volume_maps(labels, axial, min_retina)
    ends = enface.endpoints(thick, loss, across, lateral)          # CST, EZ-loss area, fovea, centring
    tier = worklist.tier(ends["decentred"], flagged, ends["central_bscans"])  # the reading priority
    cont = continuity_signal(summ, axial, limits)
    findings = {"visit": visit, "n_bscans": len(labels), "missing": missing,
                "flagged": {int(b): r for b, r in flagged.items()},
                "spacing_um": across, "axial_um": axial, "lateral_um": lateral, **ends, "tier": tier,
                # Signals: reported to the grader and logged; they do not change the tier.
                "layer_order": {"min_break_um": SEVERE_UM, "bscans_with_violations": sorted(order),
                                "violating_columns": {b: int(c.sum()) for b, c in order.items()}},
                "continuity": cont,
                "longitudinal": {"assessed": False,
                                 "reason": "no previous visit to compare with (design: Bogost et al. 2025)"}}

    # Per-B-scan lines, each with where to look: mm along the B-scan from the fovea (or the centre).
    width = labels.shape[2]
    x_mm = (np.arange(width) - (width - 1) / 2) * lateral / 1000   # column -> mm from the scan centre
    ref = "the fovea" if ends["fovea_found"] else "the scan centre"
    x_mm = x_mm - (ends["fovea_x_um"] / 1000 if ends["fovea_found"] else 0.0)
    lines = {}
    for b, s in enumerate(summ, start=1):
        if s is None:
            continue
        out = []
        if b in order:                                             # anatomy: located
            xs = x_mm[order[b]]
            out.append({"check": "layer_order", "role": "signal", "state": "fail",
                        "columns": int(order[b].sum()), "x_mm": [round(float(xs.min()), 2), round(float(xs.max()), 2)],
                        "text": f"✗ layer order broken by ≥ {SEVERE_UM:g} µm in {int(order[b].sum())} columns, "
                                f"{_mm(xs.min())} to {_mm(xs.max())} mm from {ref} along the B-scan (signal)"})
        else:
            out.append({"check": "layer_order", "role": "signal", "state": "pass",
                        "text": f"✓ no layer-order break of ≥ {SEVERE_UM:g} µm"})
        if cont["assessed"]:                                       # neighbours
            bad = b in cont["flagged_bscans"]
            out.append({"check": "continuity", "role": "signal", "state": "fail" if bad else "pass",
                        "text": "✗ jumps away from both neighbouring B-scans (signal)" if bad
                                else "✓ consistent with its neighbouring B-scans"})
        if s["retina"].any():                                      # EZ: a finding, never an error
            gone = float((s["retina"] & ~s["ez"]).sum() / s["retina"].sum())
            if gone >= EZ_FINDING_MIN:
                out.append({"check": "ez", "role": "finding", "state": "note", "ez_absent_fraction": round(gone, 3),
                            "text": f"• EZ not seen in {gone:.0%} of the retina columns: EZ loss or a "
                                    "segmentation error; check"})
        lines[b] = out

    # Visit lines: why this tier, the endpoints, and what was not assessed.
    centre = sorted(set(flagged) & set(ends["central_bscans"]))
    if tier == 1 and not ends["fovea_found"]:                     # endpoints() treats this as decentred
        why = "Priority 1: no usable retina near the centre, so the CST grid cannot be placed"
    elif tier == 1 and centre:
        why = (f"Priority 1: flagged {_bscans(centre)} {'crosses' if len(centre) == 1 else 'cross'} the central "
               "1-mm circle, where CST is measured")
    elif tier == 1:
        why = f"Priority 1: decentred, fovea {ends['fovea_offset_um']:.0f} µm from the scan centre (> 200 µm)"
    elif tier == 2:
        why = f"Priority 2: flagged {_bscans(sorted(flagged))}, none inside the central 1-mm circle"
    else:
        why = "Priority 3: nothing flagged"
    cst = ends["cst_um"]
    visit_lines = [why,
                   (f"CST {cst:.0f} µm, EZ-loss area {ends['ez_loss_mm2']:.2f} mm²" if cst is not None
                    else f"no CST: the fovea was not found; EZ-loss area {ends['ez_loss_mm2']:.2f} mm²"),
                   (f"decentration: fovea {ends['fovea_offset_um']:.0f} µm from the scan centre"
                    if ends["decentration_assessable"] and ends["fovea_found"]
                    else f"– decentration not assessable: B-scans {across:.0f} µm apart (needs ≤ 200 µm)"),
                   (f"✗ continuity: {_bscans(cont['flagged_bscans'])} "
                    f"{'jumps' if len(cont['flagged_bscans']) == 1 else 'jump'} away from the neighbours (signal)"
                    if cont["assessed"] and cont["flagged_bscans"]
                    else "✓ continuity: every B-scan consistent with its neighbours" if cont["assessed"]
                    else f"– continuity not assessed: {cont['reason']}"),
                   "– image quality not assessed: no validated gradability model (needs WRC's quality grades)",
                   "– change since the last visit not assessed: no previous visit (design: Bogost et al. 2025)"]
    if missing:
        visit_lines.insert(1, f"not acquired: {_bscans(missing)}")
    return findings, lines, visit_lines, {f"{visit}_enface_thickness.png": render_enface(thick, findings)}


def render_enface(thick: np.ndarray, findings: dict, px_per_mm: int = 60) -> bytes:
    """The thickness map as a PNG, B-scan 1 at the top, with the fovea and the central 1-mm subfield."""
    n, width = thick.shape                                         # B-scans x columns
    h_mm = (n - 1) * findings["spacing_um"] / 1000                 # map height in mm
    w_mm = width * findings["lateral_um"] / 1000                   # map width in mm
    t = np.clip((thick - SCALE_UM[0]) / (SCALE_UM[1] - SCALE_UM[0]), 0, 1)  # thickness -> 0..1
    lo, hi = np.array(RAMP[0], float), np.array(RAMP[1], float)   # the two ends of the colour ramp
    rgb = np.where(np.isnan(t)[..., None], 200.0, lo + (hi - lo) * np.nan_to_num(t)[..., None]).astype(np.uint8)
    size = (max(1, round(w_mm * px_per_mm)), max(1, round(max(h_mm, findings["spacing_um"] / 1000) * px_per_mm)))
    img = Image.fromarray(rgb).resize(size, Image.NEAREST)         # scale to mm, no blending
    d = ImageDraw.Draw(img)
    if findings["fovea_found"]:                                    # draw the 1-mm circle and a cross on the fovea
        cx = size[0] / 2 + findings["fovea_x_um"] / 1000 * px_per_mm
        cy = size[1] / 2 + findings["fovea_y_um"] / 1000 * px_per_mm
        r = enface.CENTRAL_RADIUS_UM / 1000 * px_per_mm
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 170, 0), width=2)
        d.line([cx - 5, cy, cx + 5, cy], fill=(255, 170, 0), width=2)
        d.line([cx, cy - 5, cx, cy + 5], fill=(255, 170, 0), width=2)
    d.text((4, size[1] - 14), f"thickness {SCALE_UM[0]:.0f} (light) to {SCALE_UM[1]:.0f} um (dark); b001 at top",
           fill=(0, 0, 0))                                         # the scale, written on the map
    cst = findings["cst_um"]
    d.text((4, 4), f"CST {cst:.0f} um | priority {findings['tier']}" if cst is not None else f"no CST | priority {findings['tier']}",
           fill=(0, 0, 0))                                         # the endpoint and tier on the map
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
