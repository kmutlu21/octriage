"""Volume-level checks shared by the analysis, the inference service and the triage gear:
layer order (A1), continuity between neighbouring B-scans (A2b), en face endpoints (A2) and the
visit worklist (A3). Pure numpy: no model and no Flywheel, so every check runs and is tested anywhere.
The rules are the ones pre-registered in analysis/prereg/phase_a.md (+ Amendment 1).

In plain words: Small, tested functions that look at masks without any model: are the layers in
anatomical order, does a B-scan agree with its neighbours, how thick is the retina at the centre,
and in what order should the visits be read.
"""
