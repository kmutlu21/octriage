# flywheel_gears

The two Flywheel gears, and a way to run them locally exactly as Flywheel would.

```
Flywheel: a visit's volume (NRRD), tagged "ai-ready"
  │ gear rule: relay/rule.json
  ▼
relay ── each B-scan (pixels only) ──► GPU service (../gpu_service)
  │  ◄── mask, confidence, measurements
  ▼  <visit>_relay.nrrd: every mask, every answer in its header; the input is tagged relay-done
  │ gear rule: triage/rule.json (a file named *_relay.nrrd)
triage
  ├─ each B-scan: confidence ≥ cutoff and retina found → routine; otherwise read first, with the reason
  ├─ each visit: CST, EZ-loss area, priority 1 / 2 / 3 → tag triage-priority1/2/3
  └─ <visit>_triage.txt (the checklist, in reading order) and one Flywheel QC record per check
```

| Folder | What | Start with |
|---|---|---|
| `relay/` | Gear 1: checks the volume and the service, sends every B-scan, stores the answers in one labelled volume | `run.py`, then `relay.py` |
| `triage/` | Gear 2: a generic core (cutoff, reading order, checklist) and one module per imaging type (OCT layers), so other imaging could add its own | `run.py`, then `triage.py`, `oct_layers.py` |
| `run_locally/` | Both gears under Flywheel's `flyw gear run`, on real visits, checked against the analysis | `run_visits.py` |

Each gear folder is what Flywheel needs: `manifest.json` (inputs, config, the image tag),
`Dockerfile`, `run.py` (the entry point) and `rule.json` (when Flywheel starts it; disabled until a
site switches it on). A file tagged `nogear` is skipped by both. If anything fails, a gear writes
and tags nothing and logs one line.

```bash
docker build -t local/relay:0.2.1 flywheel_gears/relay/
docker build -f flywheel_gears/triage/Dockerfile -t local/triage:0.2.1 .   # from the repo root: it copies oct_checks/
python -m flywheel_gears.run_locally.run_visits DME11                        # needs the GPU service on port 8765
```
