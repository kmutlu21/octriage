"""The visit worklist (A3): per-visit findings ranked into a reading order. In this design every
visit is still read; the worklist decides the order and points at the B-scans to check first.

In plain words: Turns each visit's results into a reading order: visits whose flagged B-scans could
change the CST endpoint first (Priority 1), then visits with flags elsewhere (2), then the rest (3).

The three reading priorities are OUR scheme, built from referenced pieces (the code calls the
level `tier`; everything a grader sees says "Priority"):
  Priority 1: the scan is decentred, or a flagged B-scan crosses the central 1-mm subfield: either can
          change the CST endpoint (FDA 2018 imaging guidance p.14: the endpoint variables are named in
          the charter; Domalpally 2009: boundary errors and decentration caused most centre-point thickness remeasurements;
          Holmen et al., JAMA Ophthalmol 2020, OCT angiography: artifact severity graded by the area of the grid affected;
          Pak 2013: the 200 µm limit).
  Priority 2: flags elsewhere only.
  Priority 3: nothing flagged.
Within a priority: more flagged B-scans first, then the visit id (a stable order).
"""
import argparse          # command-line options
import csv               # writing the CSV worklist
import html              # escaping text for the HTML page
import json              # reading the triage gear's JSON outputs
from pathlib import Path  # file paths

TIER_TEXT = {1: "Priority 1 — endpoint at risk", 2: "Priority 2 — flags outside the centre",
             3: "Priority 3 — no flags"}


def tier(decentred: bool, flagged: dict, central: list) -> int:
    """flagged: {bscan index: [reasons]}; central: B-scan indices crossing the central subfield."""
    if decentred or any(b in flagged for b in central):  # the endpoint itself may be wrong
        return 1
    return 2 if flagged else 3                           # flags elsewhere, or none at all


def rank(visits: list) -> list:
    """Give every visit its priority, then sort: priority 1 first; within a priority, most flags first."""
    for v in visits:                                                        # each visit's findings dict
        v["tier"] = tier(v["decentred"], v["flagged"], v["central_bscans"])  # add its tier
    return sorted(visits, key=lambda v: (v["tier"], -len(v["flagged"]), v["visit"]))


def _fmt(x, digits=0):
    """A number for display, or a dash when it is unknown."""
    return "—" if x is None else f"{x:.{digits}f}"


def write_csv(visits: list, path: Path) -> None:
    """The ranked worklist as a CSV, one row per visit."""
    with open(path, "w", newline="") as f:  # newline="": let the csv module write the line endings
        w = csv.writer(f)
        w.writerow(["rank", "visit", "cohort", "priority", "n_bscans", "n_flagged", "flagged_bscans", "central_bscans",
                    "decentred", "fovea_offset_um", "cst_um", "ez_loss_mm2"])  # header
        for r, v in enumerate(visits, 1):   # rank 1 = read first
            w.writerow([r, v["visit"], v.get("cohort", ""), v["tier"], v["n_bscans"], len(v["flagged"]),
                        # flagged B-scans with their reasons, e.g. "b006(confidence)"
                        " ".join(f"b{b:03d}({'+'.join(x)})" for b, x in sorted(v["flagged"].items())),
                        " ".join(f"b{b:03d}" for b in v["central_bscans"]), v["decentred"],
                        _fmt(v["fovea_offset_um"]), _fmt(v["cst_um"]), _fmt(v["ez_loss_mm2"], 3)])


def _ranges(bs: list) -> str:
    """[1, 2, 3, 7] -> 'b001–b003, b007': compact B-scan lists for the page."""
    out, bs = [], sorted(bs)             # result pieces; numbers in order
    start = prev = bs[0]                 # the current run starts at the first number
    for b in bs[1:] + [None]:            # None marks the end, to close the last run
        if b is not None and b == prev + 1:
            prev = b                     # still consecutive: extend the run
            continue
        out.append(f"b{start:03d}" if start == prev else f"b{start:03d}–b{prev:03d}")  # close the run
        if b is not None:
            start = prev = b             # a new run starts here
    return ", ".join(out)


def _flag_summary(flagged: dict) -> str:
    """Group the flagged B-scans by reason: 'confidence: b004–b006 (3)'."""
    by_reason = {}                                  # reason -> list of B-scans
    for b, reasons in flagged.items():
        for r in reasons:
            by_reason.setdefault(r, []).append(b)
    return "<br>".join(f"{html.escape(r)}: {_ranges(bs)} ({len(bs)})" for r, bs in sorted(by_reason.items())) or "—"


def _strip(v: dict) -> str:
    """One small coloured cell per B-scan, top to bottom of the volume: where the flags are."""
    cells = []
    for b in range(1, v["n_bscans"] + 1):
        kind = ("flag-central" if b in v["flagged"] and b in v["central_bscans"] else   # flagged and central
                "flag" if b in v["flagged"] else "central" if b in v["central_bscans"] else
                "missing" if b in v.get("missing", ()) else "ok")
        tip = f"b{b:03d}" + (": " + ", ".join(v["flagged"][b]) if b in v["flagged"] else "")  # hover text
        cells.append(f'<span class="c {kind}" title="{html.escape(tip)}"></span>')
    return "".join(cells)


def write_html(visits: list, path: Path, provenance: dict) -> None:
    """The ranked worklist as one self-contained HTML page (what a grader would open)."""
    rows = []
    for r, v in enumerate(visits, 1):  # one table row per visit, in reading order
        flags = _flag_summary(v["flagged"])
        rows.append(
            f'<tr class="t{v["tier"]}"><td>{r}</td><td>{html.escape(v["visit"])}</td><td>{html.escape(v.get("cohort", ""))}</td>'
            f'<td><b>{TIER_TEXT[v["tier"]]}</b>{"<br>decentred" if v["decentred"] else ""}</td>'
            f'<td class="n">{_fmt(v["cst_um"])}</td><td class="n">{_fmt(v["ez_loss_mm2"], 3)}</td>'
            f'<td class="n">{_fmt(v["fovea_offset_um"])}</td><td><div class="strip">{_strip(v)}</div></td>'
            f'<td class="f">{flags}</td></tr>')
    prov = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(str(val))}</li>" for k, val in provenance.items())
    # The page: styles (light and dark), a legend, the table, and where the results came from.
    path.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>OCT reading worklist</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --rule:#e6e5e0; --ok:#e6e5e0; --central:#9ec5f4;
        --flag:#eda100; --flagc:#b3261e; --t1:#fbe9e7; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --rule:#3a3a37;
        --ok:#3a3a37; --central:#256abf; --flag:#c98500; --flagc:#e05a4f; --t1:#3b2421; }} }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,sans-serif; margin:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} p, li {{ color:var(--ink2); }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; }}
th, td {{ border-bottom:1px solid var(--rule); padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--ink2); font-weight:600; }} td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.f {{ font-size:12px; color:var(--ink2); }} tr.t1 {{ background:var(--t1); }}
.strip {{ display:flex; flex-wrap:wrap; gap:2px; max-width:330px; }}
.c {{ width:8px; height:14px; border-radius:2px; background:var(--ok); }}
.c.central {{ background:var(--central); }} .c.flag {{ background:var(--flag); }}
.c.flag-central {{ background:var(--flagc); }} .c.missing {{ background:transparent; outline:1px dashed var(--ink2); }}
.legend .c {{ display:inline-block; vertical-align:middle; margin:0 4px 0 12px; }}
</style></head><body>
<h1>OCT reading worklist</h1>
<p>Every visit is still read; this decides the order and marks the B-scans to check first.
Priority 1 = the endpoint may be affected (decentred, or a flagged B-scan crosses the central 1-mm subfield).</p>
<p class="legend">B-scans, top to bottom of the volume:<span class="c flag-central"></span>flagged, central
<span class="c flag"></span>flagged<span class="c central"></span>central, not flagged<span class="c"></span>not flagged
<span class="c missing"></span>not in the volume</p>
<div class="wrap"><table><thead><tr><th>#</th><th>Visit</th><th>Cohort</th><th>Priority</th><th>CST µm</th>
<th>EZ loss mm²</th><th>Fovea offset µm</th><th>B-scans</th><th>Flags</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<h2 style="font-size:16px">Provenance</h2><ul>{prov}</ul>
</body></html>""", encoding="utf-8")


def load_findings(paths) -> list:
    """Triage-gear outputs (*_triage.json) back into worklist rows (JSON turned int keys into str)."""
    visits = []
    for p in paths:
        v = json.loads(Path(p).read_text(encoding="utf-8"))        # one visit's findings (✓/✗ marks: UTF-8)
        v["flagged"] = {int(b): r for b, r in v["flagged"].items()}  # B-scan numbers back to integers
        visits.append(v)
    return visits


def main():
    """Command line: rank a folder of triage-gear outputs into worklist.csv + worklist.html."""
    ap = argparse.ArgumentParser(description="Rank triage-gear outputs into a reading worklist.")
    ap.add_argument("findings", nargs="+", type=Path, help="*_triage.json files")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)                      # make the output folder if needed
    ranked = rank(load_findings(args.findings))                      # read and rank
    write_csv(ranked, args.out / "worklist.csv")
    write_html(ranked, args.out / "worklist.html", {"visits": len(ranked), "source": "triage gear outputs"})
    print(f"{len(ranked)} visits -> {args.out / 'worklist.html'}")


if __name__ == "__main__":
    main()
