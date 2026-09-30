"""The triage: which images of one visit should a grader read first, why, and where to look.

In plain words: For each image, compare the model's confidence with the cutoff (and run the OCT
checks). Flagged images go to the top of the reading order, least confident first, and every image
gets a checklist of what passed, what failed and where to look.

Pure logic with no Flywheel imports (tests/test_triage.py); flywheel_gears/triage/run.py does the Flywheel plumbing.

The core here is the same for any imaging type:
  1. the confidence decision per image: the model's own score against a cutoff chosen on data.
     The idea is WRC's tiered workflow (Domalpally et al., Ophthalmol Sci 2022;2:100198):
     an AI score plus a cutoff decides which cases a person reads first.
  2. the reading order: hard failures first, then the least confident, then the routine images.
  3. a checklist per image: every check as one line, pass (✓), fail (✗), a finding (•) or not
     assessed (–), with where to look.
Everything specific to a modality (plausibility checks, anatomy, endpoints, the visit's priority) lives
in a module; today there is one, OCT retinal layers (flywheel_gears/triage/oct_layers.py). Another imaging type
(e.g. GA on fundus autofluorescence) would add a module, not change this file.

"read first" orders human reading; it never skips anything. Whether the routine images get a full
read, a lighter check or a sample audit is set in the trial's imaging charter (FDA 2018 imaging
guidance, nonbinding, p.25: the charter defines the reading tool's role and "a subset of
computer-generated analyses" can be "verified by blinded external readers").
"""
try:
    from . import oct_layers  # imported as the triage package (tests, docs/figures.py)
except ImportError:
    import oct_layers         # inside the gear container, where the files sit side by side

MODULES = {"oct-layers": oct_layers}   # modality name (config "modality") -> its module
REASON = {"confidence": "confidence", "retina": "no retina", "result": "no result"}  # worklist labels


def fraction_from(config: dict, key: str) -> float:
    """Read a 0-1 config value. Values can arrive as strings: the flyw CLI writes numbers that way
    in local runs, and a person can hand-edit config.json, so convert and range-check here."""
    v = float(config[key])                                 # "0.9" or 0.9 -> 0.9
    if not 0.0 <= v <= 1.0:                                # a fraction must be between 0 and 1
        raise ValueError(f"{key} must be in [0, 1], got {v}")
    return v


def confidence_check(confidence: float, threshold: float) -> dict:
    """The decision on the model's own score. Inclusive: an image exactly at the cutoff passes, as
    in the analysis. Six decimals, so a score just under the cutoff never reads "x < x"."""
    ok = confidence >= threshold
    return {"check": "confidence", "role": "gate", "hard": False, "state": "pass" if ok else "fail",
            "confidence": confidence, "threshold": threshold,
            "text": f"{'✓' if ok else '✗'} confidence {confidence:.6f} {'≥' if ok else '<'} cutoff {threshold:.6f}"}


def reading_order(slices: dict) -> list[int]:
    """Read-first images come first: hard failures (e.g. no retina found), then the least confident.
    Then the routine images, least confident first. Nothing is left out."""
    first = [b for b, s in slices.items() if s["verdict"] == "read first"]
    rest = [b for b, s in slices.items() if s["verdict"] != "read first"]
    first.sort(key=lambda b: (not slices[b]["hard_failure"], slices[b]["confidence"], b))
    rest.sort(key=lambda b: (slices[b]["confidence"], b))
    return first + rest


def check_model(answers: dict, cutoff_model: str | None) -> None:
    """The confidence cutoff was chosen on one model's scores; another model's scores sit on another
    scale, so its images would be judged by the wrong line. Refuse them instead of judging silently:
    a new model needs its own cutoff (analysis/gate_analysis.py re-derives one)."""
    models = {a["model"] for a in answers.values()}
    if cutoff_model and models != {cutoff_model}:
        raise ValueError(f"the cutoff was chosen for model {cutoff_model!r}, but these answers come from "
                         f"{sorted(models)}: derive a cutoff for this model, then set cutoff_model")


def run(labels, header: dict, answers: dict, config: dict, limits: dict | None = None,
        name: str = "") -> tuple[dict, dict]:
    """labels (n, H, W) from the relay; answers {slice number: the service's answer}; config holds
    confidence_threshold, modality and the module's own settings. Returns (findings, images)."""
    module = MODULES[config.get("modality", "oct-layers")]     # KeyError for an unknown modality
    threshold = fraction_from(config, "confidence_threshold")  # the cutoff
    check_model(answers, config.get("cutoff_model"))           # the cutoff belongs to one model
    across, axial, lateral = module.spacing_um(header)         # µm per step on each axis
    slices, flagged, detail = {}, {}, {}
    for b in range(1, len(labels) + 1):                        # slice numbers are 1-based
        if module.is_missing(labels[b - 1]):
            continue                                           # not acquired: nothing to judge
        if b not in answers:                                   # a mask without an answer: never pass it
            slices[b] = {"verdict": "read first", "confidence": 0.0, "hard_failure": True,
                         "checks": [{"check": "result", "role": "gate", "hard": True, "state": "fail",
                                     "text": "✗ no model result for this image"}]}
            flagged[b], detail[b] = ["no result"], "no model result"
            continue
        c = answers[b]["confidence"]
        checks = [confidence_check(c, threshold)] + module.slice_gates(answers[b], config)
        failed = [x for x in checks if x["role"] == "gate" and x["state"] == "fail"]
        if failed:                                             # any gate check failed -> read first
            flagged[b] = [REASON[x["check"]] for x in failed]
            detail[b] = "; ".join(x["text"][2:] for x in failed)  # the text without its ✗
        slices[b] = {"verdict": "read first" if failed else "routine", "confidence": c,
                     "hard_failure": any(x.get("hard") for x in failed), "checks": checks}
    visit, more_checks, visit_lines, images = module.assess(labels, across, axial, lateral, flagged,
                                                            config, name, limits)
    for b, lines in more_checks.items():                       # signals and findings per image
        slices[b]["checks"] += lines
    findings = {**visit, "flag_detail": detail, "confidence_threshold": threshold,
                "reading_order": reading_order(slices), "visit_checks": visit_lines,
                "slices": {str(b): s for b, s in slices.items()}}
    return findings, images


def report_text(f: dict) -> str:
    """The checklist as plain text: what a grader reads in Flywheel's file viewer."""
    n_first = sum(s["verdict"] == "read first" for s in f["slices"].values())
    lines = [f"{f['visit']} · priority {f['tier']} · {n_first} of {len(f['slices'])} images to read first", ""]
    lines += f["visit_checks"] + ["", "Reading order: " + ", ".join(str(b) for b in f["reading_order"]), ""]
    for b in f["reading_order"]:
        s = f["slices"][str(b)]
        lines.append(f"Image {b} ({s['verdict']})")
        lines += [f"  {x['text']}" for x in s["checks"]]
    return "\n".join(lines) + "\n"
