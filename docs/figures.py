"""Figures for the README, the poster and the report, drawn from the repo's own data and outputs.
Every number printed on a figure is read from those files, not typed in.

In plain words: this script draws the six figures (data, messy, confidence, tiers, schedule,
grader_view), so a figure can always be redrawn from the data instead of edited by hand.

Made to a journal standard, the one WRC's own papers meet:
- Elsevier artwork guidelines (Ophthalmology Science is an Elsevier journal): widths of 90 / 140 /
  190 mm; lettering at least 7 pt at the printed size; Arial; vector PDF for line art, and images
  at >= 300 dpi (here 600).
- Rougier, Droettboom & Bourne, "Ten simple rules for better figures", PLoS Comput Biol 2014;
  10(9):e1003833: captions live in the text, not as titles in the image; panels carry letters;
  nothing that doesn't carry data.
- Colour: the dataviz reference palette's first four categorical slots in their validated order
  (blue, orange, aqua, yellow; neighbouring colours >= 9 ΔE apart under colour-vision deficiency,
  checked with its validator). The four retinal bands are neighbours, so they take that order.

    python docs/figures.py                      # every figure -> docs/figures/<name>.svg and .png (600 dpi)
    python docs/figures.py --only schedule      # one figure

One layout serves the report and the poster: the poster places the same vector figure (SVG) at
>= 2.6x its journal width, so 7-pt lettering prints at >= 18 pt and every label keeps its place.
Scans are named for readers in the captions ("DME patient 11, B-scan 6 of 19"); file ids in notes.
"""
import argparse                          # command-line options
import sys                               # the import path
from pathlib import Path                 # file paths

import matplotlib

matplotlib.use("Agg")                    # draw to files, no screen needed
import matplotlib.pyplot as plt          # figures
import numpy as np                       # arrays
import pandas as pd                      # tables
import torch                             # running the model for the confidence figure
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # colour helpers
from matplotlib.patches import Patch     # legend entries for filled areas
from PIL import Image                    # reading images and masks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # so `gpu_service`, `oct_checks`, `flywheel_gears` import when run as a script
from analysis.gate_analysis import bootstrap  # noqa: E402  the patient bootstrap
from gpu_service.unet.unet import load_checkpoint, preprocess  # noqa: E402
from oct_checks.enface import CENTRAL_RADIUS_UM  # noqa: E402  the 1-mm circle's radius
from oct_checks.geometry import AXIAL_UM, LATERAL_UM, bscan_spacing_um, en_face_coords  # noqa: E402
from oct_checks.worklist import TIER_TEXT  # noqa: E402  what the worklist prints for each priority
from gpu_service.tta import AUGMENTS, apply, invert  # noqa: E402
from flywheel_gears.triage import oct_layers  # noqa: E402  the triage gear's own OCT logic

EXP = REPO / "outputs/unet_w32_cv5_ez_border"          # the served model's cross-validation run
DATA = REPO / "data"                                   # OCT5k + the manifest
WEIGHTS = REPO / "weights/unet_w32_cv5_ez_border-fold0.pt"  # the served model (fold 0)
PHASE_A = EXP / "analysis/phase_a/amended"             # the visit-level results
MM = 1 / 25.4                                          # inches per millimetre
OUT = Path(__file__).resolve().parent / "figures"     # where the figures go
FORMATS = ("svg", "png")                               # SVG for the poster, PNG for the README

# Colours (dataviz reference palette, light): ink for text; categorical slots 1-4 in validated order.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BAND = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW}       # the four retinal bands, top to bottom (neighbours)
BAND_NAME = {1: "ILM–OPL", 2: "OPL–IS/OS", 3: "EZ", 4: "RPE"}
GRADER = {1: BLUE, 2: ORANGE, 3: AQUA}                # three graders: slots 1-3 pass all pairs
RED = "#d03b3b"                                        # "look here" pixels (status critical)
READ_FIRST = ORANGE                                    # everything "read first", in every figure
TIER = {1: "#d03b3b", 2: "#fab219", 3: "#0ca30c"}      # reading priorities = status colours (critical /
                                                       # warning / good), always beside "Priority N"
PHASE_C = EXP / "analysis/phase_c"                     # the reading schedule (pre-registered Phase C)
PIPE = REPO / "outputs/pipeline_demo"                  # the gears' own outputs on four real visits
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "svg.fonttype": "none",   # real, editable text in PDF and SVG
})

scans = pd.read_csv(EXP / "analysis/scans.csv").set_index("scan_id")       # one row per B-scan
manifest = pd.read_csv(DATA / "manifest.csv").set_index("scan_id")        # file paths per B-scan
cutoff = float(pd.read_json(EXP / "analysis/results.json", typ="series")["deployment_threshold_all_folds"])
N_BSCANS = manifest.groupby("patient").bscan.max()   # B-scans in each volume (the highest B-scan number)


# ---------- helpers ----------
def size(w_mm: float, h_mm: float) -> tuple[float, float]:
    """Figure size in inches from millimetres."""
    return w_mm * MM, h_mm * MM


def save(fig, name: str) -> None:
    """Every format: vector PDF and SVG for print and the poster, PNG at 600 dpi."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print("wrote", name, "->", OUT)


def letter(ax, s: str, dx: float = -2, dy: float = 3) -> None:
    """A panel letter, bold, just outside the panel's top-left corner."""
    ax.annotate(s, (0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                ha="right", va="bottom", fontsize=10, fontweight="bold", color=INK)


def label(ax, text: str, x: float = 0.015, y: float = 0.97, **kw) -> None:
    """A short label inside an image panel, on a white box so it reads over any image."""
    ax.text(x, y, text, transform=ax.transAxes, ha=kw.pop("ha", "left"), va=kw.pop("va", "top"), fontsize=7,
            color=INK, bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.9), **kw)


def raw(scan_id: str) -> np.ndarray:
    """The source B-scan as stored (512 rows x 496 columns)."""
    return np.array(Image.open(DATA / manifest.loc[scan_id, "image"]).convert("L"))


def image(scan_id: str) -> np.ndarray:
    """The B-scan as the model sees it: 512 x 512, same grid as the masks."""
    return np.array(Image.fromarray(raw(scan_id)).resize((512, 512), Image.BILINEAR))


def boundaries(scan_id: str, grader: int) -> np.ndarray:
    """(512 columns, 5 boundaries) rows of ILM, OPL, IS-OS, IBRPE, OBRPE for one grader."""
    return np.loadtxt(DATA / manifest.loc[scan_id, f"boundary_g{grader}"], delimiter=",", skiprows=1)[:, 1:]


def retina_rows(mask: np.ndarray, margin: int = 40) -> tuple[int, int]:
    """First and last image row holding retina (bands 1-4), plus a margin: used to crop."""
    rows = np.flatnonzero(np.isin(mask, (1, 2, 3, 4)).any(1))  # rows with any retinal band
    return max(rows.min() - margin, 0), min(rows.max() + margin, 511)


def tint(color: str, share: float) -> tuple:
    """`color` laid over white at `share`, as an OPAQUE colour. Transparent hatch fills are written to
    SVG as fill-opacity on a pattern, which Chromium's PDF printer ignores (the region prints solid)."""
    return tuple(1 - share * (1 - c) for c in to_rgb(color))


def bare(ax):
    """No ticks, no frame: for images."""
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def overlay(ax, img, mask, bands=(1, 2, 3, 4), alpha=0.45):
    """The B-scan in grey with the mask's bands painted on top, semi-transparent."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    rgba = np.zeros((*mask.shape, 4))                  # a transparent layer
    for b in bands:
        rgba[mask == b] = (*to_rgb(BAND[b]), alpha)    # colour each band's pixels
    ax.imshow(rgba)
    bare(ax)


def legend_below(fig, handles: list, ncol: int) -> None:
    """A legend centred just under the lowest panel (not at a fixed figure position)."""
    fig.canvas.draw()                                            # settle the layout first
    r = fig.canvas.get_renderer()                                # the lowest panel, labels included
    bottom = min(fig.transFigure.inverted().transform(a.get_tightbbox(r).get_points())[0, 1] for a in fig.axes)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, bottom - 0.015), ncol=ncol,
               handlelength=1.2, columnspacing=1.2)


def band_handles() -> list:
    """Legend entries for the four band colours."""
    return [Patch(color=BAND[b], label=BAND_NAME[b]) for b in BAND]


# ---------- figures ----------
def draw_volume(ax, patient: str = "DME28", highlight: int = 12):
    """One visit as a stack of its B-scans, B-scan 1 in front; `highlight` outlined."""
    ids = manifest[manifest.patient == patient].sort_values("bscan").index.tolist()  # its B-scans in order
    crops = [image(s)[120:420] for s in ids]  # the retinal band of these scans
    w, h, dx, dy = 512, 300, 26, 20           # crop size; offset of each B-scan behind the previous
    for i in reversed(range(len(crops))):     # back first, so the front B-scan is drawn on top
        x0, y0 = i * dx, i * dy
        ax.imshow(crops[i], cmap="gray", vmin=0, vmax=255, extent=(x0, x0 + w, y0, y0 + h), zorder=len(crops) - i)
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="#9aa4b2", lw=0.4, zorder=len(crops) - i))
    n = len(crops)
    if highlight:                              # the B-scan shown in panel B, outlined on top of the stack
        x0, y0 = (highlight - 1) * dx, (highlight - 1) * dy
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=INK, lw=1.2, zorder=n + 1))
        ax.text(x0 + w + 8, y0 + h - 8, f"B-scan {highlight}", ha="left", va="top", fontsize=7, color=INK,
                zorder=n + 2, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(-10, w + (n - 1) * dx + 10)
    ax.set_ylim(-60, h + (n - 1) * dy + 60)
    ax.text(0, -12, "B-scan 1", va="top", fontsize=7, color=INK2)
    ax.text((n - 1) * dx + w, (n - 1) * dy + h + 12, f"B-scan {n}", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_aspect("equal")
    bare(ax)


def draw_graders(ax):
    """Three graders' five boundaries on one hard B-scan (DME patient 28, B-scan 12)."""
    sid = "DME28_b012"
    b = {g: boundaries(sid, g) for g in (1, 2, 3)}      # each grader's 5 boundary lines
    ax.imshow(image(sid), cmap="gray", vmin=0, vmax=255)
    x = np.arange(512)
    for g in (1, 2, 3):
        for j in range(5):
            ax.plot(x, b[g][:, j], color=GRADER[g], lw=0.8, label=f"Grader {g}" if j == 0 else None)
    lo = min(v.min() for v in b.values()) - 25          # crop to the traced region
    hi = max(v.max() for v in b.values()) + 25
    ax.set_ylim(hi, lo)
    ax.set_aspect("auto")
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.5)
    bare(ax)


def data():
    """Patient DME-28. A: the visit is a volume of 19 B-scans. B: B-scan 12's three graders."""
    fig = plt.figure(figsize=size(190, 64))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.06)
    ax = fig.add_subplot(gs[0])
    draw_volume(ax)
    letter(ax, "A", dx=0)
    ax = fig.add_subplot(gs[1])
    draw_graders(ax)
    letter(ax, "B")
    save(fig, "data")


def messy():
    """What looking at the data found: A a noise frame, B burned-in text, C crossing boundaries."""
    fig, axes = plt.subplots(1, 3, figsize=size(190, 60), gridspec_kw={"wspace": 0.06})
    axes[0].imshow(image("Normal38_b001"), cmap="gray", vmin=0, vmax=255)
    axes[1].imshow(image("AMD10_b001")[:200], cmap="gray", vmin=0, vmax=255)       # top 200 rows only
    sid = "DME11_b007"  # the grading with the widest crossing in OCT5k (191 columns, up to 10 px)
    img, b = image(sid), boundaries(sid, 1)
    cross = b[:, 3] < b[:, 2]                     # IBRPE drawn above IS-OS in these columns
    c = int(np.argmax(b[:, 2] - b[:, 3]))         # the deepest crossing
    x0, x1 = max(c - 60, 0), min(c + 60, 511)     # zoom onto 120 columns around it
    lo, hi = int(b[x0:x1, 2:4].min()) - 18, int(b[x0:x1, 2:4].max()) + 18
    x = np.arange(512)
    axes[2].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[2].fill_between(x, b[:, 2], b[:, 3], where=cross, color=RED, alpha=0.5, lw=0, label="Crossed")
    axes[2].plot(x, b[:, 2], color=BLUE, lw=1.2, label="IS-OS (top of EZ)")
    axes[2].plot(x, b[:, 3], color=ORANGE, lw=1.2, label="IBRPE (bottom of EZ)")
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(hi, lo)
    axes[2].legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.2)
    for ax, s in zip(axes, "ABC"):
        ax.set_aspect("auto")                     # three panels of equal size; the zoom is stretched
        bare(ax)
        letter(ax, s)
    save(fig, "messy")


@torch.no_grad()
def tta_passes(sid: str):
    """The 6 test-time passes of the served model, each mapped back onto the original scan."""
    model, _ = load_checkpoint(WEIGHTS, "cpu")
    x = preprocess(raw(sid))
    copies = [apply(x, f, s, model.shift_mode) for f, s in AUGMENTS]           # the 6 altered copies
    probs = model(torch.cat(copies)).softmax(1)                                 # 6 predictions at once
    probs = torch.stack([invert(p, f, s, model.shift_mode) for p, (f, s) in zip(probs, AUGMENTS)])  # undo each
    return [c[0, 0].numpy() for c in copies], probs.argmax(1).numpy(), probs.mean(0).argmax(0).numpy()


def confidence():
    """A: the six copies of one B-scan. B, C: where their six masks disagree, routine vs read first."""
    names = ["As is", "Flipped", "Up 8 px", "Down 8 px", "Flipped, up", "Flipped, down"]
    picks = [("DME30_b006", "Routine"), ("DME11_b006", "Read first")]
    fig = plt.figure(figsize=size(190, 104))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 2.0], hspace=0.30, wspace=0.05)
    copies, _, _ = tta_passes(picks[1][0])
    for i, (c, text) in enumerate(zip(copies, names)):       # top row: the 6 copies
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(c[100:460], cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel(text, fontsize=7, color=INK, labelpad=2)
        bare(ax)
        if i == 0:
            letter(ax, "A")
    for k, (sid, verdict) in enumerate(picks):                # bottom row: where the 6 masks disagree
        _, passes, consensus = tta_passes(sid)
        retina = np.isin(passes, (1, 2, 3, 4)).any(0)
        disagree = (passes != consensus[None]).any(0) & retina   # any pass differs from the consensus
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
        overlay(ax, image(sid), consensus, alpha=0.25)
        rgba = np.zeros((512, 512, 4))
        rgba[disagree] = (*to_rgb(RED), 1.0)
        ax.imshow(rgba)
        top, bottom = retina_rows(consensus, 30)
        ax.set_ylim(bottom, top)
        ax.set_aspect("auto")
        c = scans.loc[sid, "confidence"]
        label(ax, f"{verdict}: confidence {c:.4f} {'≥' if c >= cutoff else '<'} {cutoff:.4f}\n"
                  f"{disagree.sum():,} pixels disagree")
        letter(ax, "BC"[k])
    legend_below(fig, band_handles() + [Patch(color=RED, label="Six masks disagree")], ncol=5)
    save(fig, "confidence")


def enface_map(ax, patient: str) -> dict:
    """One visit's thickness map, top-down, to scale in mm, as the triage gear computes it:
    flagged B-scans as lines, the central 1-mm circle (where CST is read) around the fovea."""
    n = int(N_BSCANS[patient])                                      # B-scans in the volume
    sub = scans[scans.patient == patient].join(manifest.bscan)      # its B-scans, with their numbers
    vol = np.full((n, 512, 512), oct_layers.MISSING, np.uint8)      # positions with no mask stay "missing"
    flagged = {}
    for sid, r in sub.iterrows():
        vol[r.bscan - 1] = np.array(Image.open(EXP / r.pred_mask))  # the model's mask at its position
        if not r.passed:
            flagged[int(r.bscan)] = ["confidence"]                  # read first
    spacing = bscan_spacing_um(n)                                   # µm between neighbouring B-scans
    f, _, _, _ = oct_layers.assess(vol, spacing, AXIAL_UM, LATERAL_UM, flagged, {}, patient, None)
    thick = oct_layers.volume_maps(vol, AXIAL_UM, 0.9)[3]          # the same thickness map the gear draws
    y, x = en_face_coords(n, 512, spacing)                          # µm positions of rows and columns
    half = spacing / 2 / 1000                                       # each B-scan row covers +- half a gap
    extent = (x[0] / 1000, x[-1] / 1000, y[-1] / 1000 + half, y[0] / 1000 - half)  # B-scan 1 at the top
    ramp = LinearSegmentedColormap.from_list("thick", [np.array(c) / 255 for c in oct_layers.RAMP])
    im = ax.imshow(thick, cmap=ramp, vmin=oct_layers.SCALE_UM[0], vmax=oct_layers.SCALE_UM[1], extent=extent,
                   aspect="equal", interpolation="nearest")
    for b in f["flagged"]:                                          # every flagged B-scan: a line across the map
        ax.axhline(y[b - 1] / 1000, color=READ_FIRST, lw=1.4)
    if f["fovea_found"]:                                            # the ETDRS grid, centred on the fovea
        fx, fy = f["fovea_x_um"] / 1000, f["fovea_y_um"] / 1000
        for r in (1.5, 3.0):                                        # the 3- and 6-mm circles (Pak 2013)
            ax.add_patch(plt.Circle((fx, fy), r, fill=False, ec=INK2, lw=0.6))
        for a in np.deg2rad([45, 135, 225, 315]):                   # the spokes that split the rings in four
            ax.plot([fx + 0.5 * np.cos(a), fx + 3.0 * np.cos(a)], [fy + 0.5 * np.sin(a), fy + 3.0 * np.sin(a)],
                    color=INK2, lw=0.6)
        ax.add_patch(plt.Circle((fx, fy), CENTRAL_RADIUS_UM / 1000, fill=False, ec=INK, lw=1.3))  # 1 mm: CST
        ax.plot(fx, fy, "+", color=INK, ms=6, mew=1.2)              # the fovea
    ax.set_xlim(-4.45, 4.45)                                        # every map in the same 8.9 mm frame
    ax.set_ylim(3.95, -3.95)                                        # 7.4 mm plus room for the edge B-scans
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_color(INK2), s.set_linewidth(0.5)
    f["_image"] = im
    return f


def tiers():
    """The priority rule on three real visits, one per priority: thickness maps with the ETDRS grid.
    All three are DME eyes with 19 B-scans, so the maps compare directly. Priority 2 visits never
    have many flags (at most 4); DME14 has 2, and one of them is a real poor mask."""
    examples = [("DME11", 1), ("DME14", 2), ("DME30", 3)]
    fig = plt.figure(figsize=size(190, 80))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.08)
    for k, (patient, tier) in enumerate(examples):
        ax = fig.add_subplot(gs[0, k])
        f = enface_map(ax, patient)
        for s in ax.spines.values():                                # the priority's colour frames its map
            s.set_color(TIER[tier]), s.set_linewidth(2.0)
        ax.set_title(f"Priority {tier}", fontsize=8, pad=3, fontweight="bold")
        ax.set_xlabel(f"{len(f['flagged'])} of {f['n_bscans']} B-scans flagged · CST {f['cst_um']:.0f} µm",
                      fontsize=7, color=INK2, labelpad=3)
        letter(ax, "ABC"[k])
    cax = fig.add_subplot(gs[0, 3])                                 # the colour scale
    cb = fig.colorbar(f["_image"], cax=cax)
    cb.set_label("Retinal thickness (µm)", fontsize=7)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2)
    cb.outline.set_linewidth(0.5)
    fig.canvas.draw()                                               # maps keep their aspect: match their height
    m, c = fig.axes[0].get_position(), cax.get_position()
    cax.set_position([c.x0, m.y0, c.width, m.height])
    legend_below(fig, [plt.Line2D([], [], color=READ_FIRST, lw=1.4, label="Flagged B-scan"),
                       plt.Line2D([], [], color=INK, lw=1.3, marker="+", ms=6, label="1-mm centre, where CST is read"),
                       plt.Line2D([], [], color=INK2, lw=0.6, label="ETDRS 3- and 6-mm rings")], ncol=3)
    save(fig, "tiers")


def schedule():
    """The schedule a grader would follow (Phase C, C1): whole visits in priority order, the gear's reading
    order inside each; poor masks found against B-scans read, with random and perfect order."""
    sch = pd.read_csv(PHASE_C / "schedule.csv")
    vis = pd.read_csv(PHASE_C / "visits_schedule.csv")
    n, total = len(sch), int(sch.error.sum())
    x = np.concatenate([[0], sch.read.to_numpy()])
    found = np.concatenate([[0], sch.poor_found.to_numpy()])
    end = {t: int(sch.read[sch.tier <= t].max()) for t in (1, 2)}              # B-scans read after priorities 1, 2
    fig, ax = plt.subplots(figsize=size(140, 92))
    fig.subplots_adjust(bottom=0.27)                                           # room for the legend below
    for t, (a, b) in zip((1, 2, 3), ((0, end[1]), (end[1], end[2]), (end[2], n))):
        ax.axvspan(a, b, facecolor=tint(TIER[t], 0.13), lw=0, zorder=0)       # each priority's stretch of reading
        nv = int((vis.tier == t).sum())
        ax.text((a + b) / 2, total + 5.6, f"Priority {t}", ha="center", va="top", fontsize=7, color=INK,
                fontweight="bold")
        ax.text((a + b) / 2, total + 3.3, f"{nv} visits", ha="center", va="top", fontsize=7, color=INK2)
    ax.plot(x, np.minimum(x, total), color=INK, lw=1.8, ls=(0, (4, 2)), zorder=2)             # perfect order
    ax.plot(x, total * x / n, color=INK2, lw=1.6, ls=(0, (1, 1.6)), zorder=2)                 # random order
    ax.plot(x, found, color=BLUE, lw=3.2, zorder=3, solid_capstyle="round")                   # OCTriage
    cst = [int(sch.read[sch.visit_rank == r].max()) for r in vis["rank"][vis.cst_error.fillna(False).astype(bool)]]
    ax.scatter(cst, found[cst], marker="v", s=30, color=TIER[1], edgecolor="white", lw=0.6, zorder=5)
    # The gain over a random order at the end of Priority 1: a double arrow from random to OCTriage.
    k1 = end[1]
    rand = total * k1 / n                                                      # poor masks a random order finds
    ax.scatter([k1], [found[k1]], s=26, color=BLUE, edgecolor="white", lw=1.0, zorder=4)
    ax.annotate("", xy=(k1, found[k1] - 1.2), xytext=(k1, rand + 0.4),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0))
    assert found[k1] > 2 * rand                                                # the label below must stay true
    ax.text(k1 - 25, 31.0, "More than double\na random order", ha="right", va="center", fontsize=7,
            fontweight="bold", color=INK)                                      # in the empty band between the curves
    ax.text(k1 - 25, 24.6, f"{found[k1]} vs about {rand:.0f} poor masks\nafter Priority 1 ({k1 / n:.0%} read)",
            ha="right", va="center", fontsize=7, color=INK)
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=3.2, label="OCTriage: whole visits, priority order"),
                       plt.Line2D([], [], color=INK, lw=1.8, ls=(0, (4, 2)), label="Perfect order"),
                       plt.Line2D([], [], color=INK2, lw=1.6, ls=(0, (1, 1.6)), label="Random order (expected)"),
                       plt.Line2D([], [], color=TIER[1], lw=0, marker="v", ms=5, label="A visit with a CST error")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=2.0, columnspacing=2.0)
    ax.set(xlabel=f"B-scans read (of {n:,})", ylabel=f"Poor masks found (of {total})", xlim=(0, n),
           ylim=(0, total + 6))
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(color=GRID, lw=0.4, axis="y")
    ax.set_axisbelow(True)
    save(fig, "schedule")


def grader_view():
    """What a grader would see in Flywheel (a schematic built from the gears' real outputs, not a
    screenshot), top to bottom: the worklist, patient DME-11's B-scans in reading order, and two of
    them close up with the gear's checklist. Lettering is 8.5 pt, a step up from the other figures:
    this one carries a table and a checklist, and the poster shows it in its narrower middle column."""
    vis = pd.read_csv(PHASE_C / "visits_schedule.csv").sort_values("rank")
    flags = pd.read_csv(PHASE_A / "bscans.csv").groupby("patient").review_final.sum()
    cst = pd.read_csv(PHASE_A / "visits.csv").set_index("visit").cst_model
    f = pd.read_json(PIPE / "DME11/DME11_triage.json", typ="series")
    order, sl = list(f["reading_order"]), f["slices"]
    FS = 8.5                                                         # lettering size in this figure, pt
    fig = plt.figure(figsize=size(190, 232))                        # tall: fills the poster's middle column
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.75, 1.3], hspace=0.22)

    # A: the worklist, one row per visit, in reading order (the first five, two of Priority 2, one of 3)
    ax = fig.add_subplot(gs[0])
    bare(ax)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    cols = (0.0, 0.42, 0.58, 0.79)                                 # Priority, Patient, Flagged, CST
    for x, head in zip(cols, ("Priority", "Patient", "Flagged B-scans", "CST (µm)")):
        ax.text(x, 0.985, head, fontsize=FS, fontweight="bold", color=INK2, va="top")
    ax.plot([0, 1], [0.9, 0.9], color=INK2, lw=0.6)
    rows = (list(vis.head(5).itertuples()) + [None] + list(vis[vis.tier == 2].head(2).itertuples()) + [None]
            + list(vis[vis.tier == 3].head(1).itertuples()))
    step = 0.86 / (sum(r is not None for r in rows) + 0.7 * rows.count(None))  # rows fill the panel
    y = 0.9 - 0.6 * step
    for r in rows:
        if r is None:                                                # a gap in the list
            ax.text(cols[1], y + 0.25 * step, "⋮", fontsize=FS + 1, color=INK2, va="center",
                    fontfamily="DejaVu Sans")
            y -= 0.7 * step
            continue
        if r.visit == "DME11":                                      # the visit opened below
            ax.add_patch(plt.Rectangle((-0.005, y - 0.48 * step), 1.01, 0.96 * step, color="#ecebe6", lw=0,
                                       zorder=0))
        ax.add_patch(plt.Rectangle((0.0, y - 0.28 * step), 0.012, 0.56 * step, color=TIER[r.tier], lw=0))
        bold = "bold" if r.visit == "DME11" else None
        ax.text(0.024, y, TIER_TEXT[r.tier], fontsize=FS, color=INK, va="center")
        ax.text(cols[1], y, f"{r.visit[:-2]}-{r.visit[-2:]}", fontsize=FS, color=INK, va="center",
                fontweight=bold)
        ax.text(cols[2], y, f"{int(flags[r.visit])} of {r.n_bscans}", fontsize=FS, color=INK, va="center")
        ax.text(cols[3], y, f"{cst[r.visit]:.0f}", fontsize=FS, color=INK, va="center")
        y -= step
    letter(ax, "A", dy=-8)

    # B: the visit's B-scans in reading order: flagged first, then the rest
    first = [b for b in order if sl[str(b)]["verdict"] == "read first"]
    rest = [b for b in order if sl[str(b)]["verdict"] != "read first"]
    sub = gs[1].subgridspec(2, 12, hspace=0.62, wspace=0.08)
    for row, (bs, title) in enumerate(((first, f"Read first: the {len(first)} flagged, least confident first"),
                                       (rest, f"Then the other {len(rest)}"))):
        for i, b in enumerate(bs):
            ax = fig.add_subplot(sub[row, i])
            ax.imshow(image(f"DME11_b{b:03d}")[100:460], cmap="gray", vmin=0, vmax=255, aspect="auto")
            bare(ax)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(READ_FIRST if row == 0 else INK2), sp.set_linewidth(1.4 if row == 0 else 0.5)
            ax.set_xlabel(str(b), fontsize=FS, color=INK, labelpad=1,
                          fontweight="bold" if b in (6, 13) else None)
            if i == 0:
                ax.set_title(title, fontsize=FS, color=INK, loc="left", pad=2)
                if row == 0:
                    ax.annotate(f"Patient DME-11 · {f['visit_checks'][0]}", (0, 1), xycoords="axes fraction",
                                xytext=(0, 15), textcoords="offset points", fontsize=FS, fontweight="bold",
                                color=INK, va="bottom")
                    letter(ax, "B", dy=15)

    # C: two B-scans close up, each with the gear's checklist (shortened)
    sub = gs[2].subgridspec(2, 2, height_ratios=[1.0, 0.85], wspace=0.1, hspace=0.06)
    checklists = []
    for k, b in enumerate((6, 13)):
        s = sl[str(b)]
        mask = np.array(Image.open(EXP / "oof_masks" / f"DME11_b{b:03d}.png"))
        ax = fig.add_subplot(sub[0, k])
        overlay(ax, image(f"DME11_b{b:03d}"), mask, alpha=0.35)
        top, bottom = retina_rows(mask, 30)
        ax.set_ylim(bottom, top)
        ax.set_aspect("auto")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(READ_FIRST if s["verdict"] == "read first" else INK2), sp.set_linewidth(1.4)
        pos = order.index(b) + 1
        ax.set_title(f"B-scan {b}: {'read first' if s['verdict'] == 'read first' else 'routine'}, "
                     f"{pos}{'st' if pos == 1 else 'th'} in order", fontsize=FS, color=INK, loc="left", pad=2)
        if k == 0:
            letter(ax, "C")
        lines = []
        for c in s["checks"]:                                        # the gear's own checklist, shortened
            if c["check"] == "confidence":
                ok = c["state"] == "pass"
                lines.append((ok, f"confidence {s['confidence']:.3f} {'≥' if ok else '<'} cutoff "
                                  f"{f['confidence_threshold']:.3f}"))
            elif c["check"] == "retina":
                lines.append((c["state"] == "pass", "retina found"))
            elif c["check"] == "layer_order" and c["state"] == "fail":
                lo, hi = c["x_mm"]
                lines.append((False, f"layer order broken (warning),\n{lo:+.1f} to {hi:+.1f} mm from the fovea"))
            elif c["check"] == "continuity" and c["state"] in ("pass", "fail"):
                lines.append((c["state"] == "pass", "consistent with its neighbouring B-scans"
                              if c["state"] == "pass" else "jumps away from its neighbouring B-scans (warning)"))
            elif c["check"] == "ez":
                lines.append((None, f"EZ not seen in {c['ez_absent_fraction']:.0%}: loss or error?"))
        checklists.append((sub[1, k], lines))
    n_lines = max(sum(t.count("\n") + 1 for _, t in lines) for _, lines in checklists)
    dy = 0.96 / n_lines                                              # one line pitch for both checklists
    for cell, lines in checklists:
        tx = fig.add_subplot(cell)
        bare(tx)
        tx.set_xlim(0, 1), tx.set_ylim(0, 1)
        y = 0.98
        for ok, text in lines:
            icon, col = ("✓", TIER[3]) if ok else (("✗", TIER[1]) if ok is False else ("•", INK2))
            tx.text(0.0, y, icon, fontsize=FS + 1, color=col, va="top", fontfamily="DejaVu Sans")
            tx.text(0.045, y, text, fontsize=FS, color=INK, va="top", linespacing=1.25)
            y -= dy * (text.count("\n") + 1)
    save(fig, "grader_view")


FIGURES = {f.__name__: f for f in (data, messy, confidence, tiers, schedule, grader_view)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="draw only these figures")
    args = ap.parse_args()
    for key in args.only or FIGURES:                       # all figures unless --only
        FIGURES[key]()
