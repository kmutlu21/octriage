"""The manifests, the gear rules and the gear code must agree; nothing else checks this before upload."""
import json              # reading manifests and rules
from pathlib import Path  # file paths

import jsonschema        # validating against Flywheel's official schema
import pytest            # the test framework

ROOT = Path(__file__).resolve().parents[1]
GEARS = ("relay", "triage")
MANIFEST = {g: json.loads((ROOT / "flywheel_gears" / g / "manifest.json").read_text()) for g in GEARS}  # what each gear declares
RULE = {g: json.loads((ROOT / "flywheel_gears" / g / "rule.json").read_text()) for g in GEARS}          # when Flywheel runs it

# From flywheel-sdk 22.5.0: models/gear_rule_input.py and models/gear_rule_condition_type.py.
RULE_FIELDS = {"project_id", "gear_id", "role_id", "name", "config", "fixed_inputs", "priority",
               "auto_update", "any", "all", "not", "disabled", "compute_provider_id",
               "triggering_input", "tags"}
CONDITION_TYPES = {"file.type", "file.name", "file.modality", "file.classification", "file.tags",
                   "file.parent_ref.type", "container.has-type", "container.has-classification",
                   "gear.name", "gear.version"}


@pytest.mark.parametrize("gear", GEARS)
def test_manifest_matches_official_schema(gear):
    schema = json.loads((ROOT / "tests/data/manifest.schema.json").read_text())
    jsonschema.validate(MANIFEST[gear], schema)                  # raises if anything is malformed


@pytest.mark.parametrize("gear", GEARS)
def test_rule_uses_only_known_fields_and_conditions(gear):
    assert set(RULE[gear]) <= RULE_FIELDS                         # no field the SDK doesn't know
    for cond in RULE[gear]["all"] + RULE[gear]["any"] + RULE[gear]["not"]:
        assert cond["type"] in CONDITION_TYPES


@pytest.mark.parametrize("gear", GEARS)
def test_rule_triggers_the_gears_input_with_valid_config(gear):
    rule, man = RULE[gear], MANIFEST[gear]
    assert rule["triggering_input"] in man["inputs"]              # the rule feeds a real input
    assert set(rule["config"]) <= set(man["config"])              # no setting the gear doesn't have
    for key, value in rule["config"].items():
        spec = man["config"][key]
        assert isinstance(value, {"string": str, "number": (int, float)}[spec["type"]])  # right type
        assert spec.get("minimum", value) <= value <= spec.get("maximum", value)        # in range
        assert value in spec.get("enum", [value])                                         # allowed value
        if "default" in spec and key != "service_url":
            assert spec["default"] == value                       # manifest default = what the rule sets


def test_rules_cannot_retrigger_on_their_own_outputs():
    relay_not = {c["value"] for c in RULE["relay"]["not"] if c["type"] == "file.tags"}
    assert RULE["relay"]["config"]["done_tag"] in relay_not and "nogear" in relay_not  # Orchestra-Gear's skip tag
    triage_not = {c["value"] for c in RULE["triage"]["not"] if c["type"] == "file.tags"}
    prefix = RULE["triage"]["config"]["tag_prefix"]
    assert {f"{prefix}{t}" for t in (1, 2, 3)} <= triage_not and "nogear" in triage_not


def test_triage_starts_on_what_the_relay_writes():
    import re
    pattern = next(c["value"] for c in RULE["triage"]["all"] if c["type"] == "file.name")
    assert re.search(pattern, "DME11_relay.nrrd") and not re.search(pattern, "DME11.nrrd")


# The served model's analysis output (gitignored; the test skips on a fresh clone).
ANALYSIS = ROOT / "outputs/unet_w32_cv5_ez_border/analysis/results.json"


def test_triage_cutoff_is_the_analysis_cutoff_unrounded():
    # The Youden optimum sits just above an error, so rounding it down passes that error in the gear
    # while the analysis reads it first. A real bug we had: an earlier gear rounded 0.98892652 to
    # 0.9889 and passed a scan (0.9889243) the analysis flags. Today's 0.98760415 -> 0.987 would pass
    # AMD03_b036 (0.9875404).
    if not ANALYSIS.exists():
        pytest.skip("analysis outputs not present")
    exact = json.loads(ANALYSIS.read_text())["deployment_threshold_all_folds"]
    assert MANIFEST["triage"]["config"]["confidence_threshold"]["default"] == \
        RULE["triage"]["config"]["confidence_threshold"] == exact
    assert not 0.9875403694413948 >= exact  # AMD03_b036 is read first
