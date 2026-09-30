"""The pre-registered score comparison: the rule file is unchanged since it was frozen,
AURC behaves as defined, and the decision rule picks what the pre-registration says it picks."""
import hashlib           # sha256 of the rule file

import numpy as np       # arrays
import pandas as pd      # tables
import pytest            # the test framework

from analysis import confidence_eval as ce  # the code under test


def test_preregistration_is_unchanged_since_it_was_frozen():
    assert hashlib.sha256(ce.PREREG.read_bytes()).hexdigest() == ce.PREREG_SHA256


def test_aurc_rewards_ranking_the_risky_scans_last():
    risk = np.array([0.0, 0.0, 0.1, 0.5])
    perfect = ce.aurc(-risk, risk)                                     # confidence = minus the true risk
    assert perfect == pytest.approx(ce.aurc(risk, risk, "opt-aurc"))  # = Zenk's "optimal" AURC
    assert ce.aurc(risk, risk) > ce.aurc([4, 3, 2, 1], risk) == pytest.approx(perfect)  # backwards is worse
    assert ce.aurc([4, 3, 2, 1], risk, "norm-aurc") == pytest.approx(1.0)


def test_review_takes_the_least_confident_with_deterministic_ties():
    conf = np.array([0.9, 0.5, 0.5, 0.1, 0.8])
    ids = np.array(["e", "b", "a", "d", "c"])
    passed = ce.passed_at_review(conf, ids, 0.4)  # round(0.4 * 5) = 2 reviewed
    assert passed.tolist() == [True, False, True, False, True]  # of the tied 0.5s, "a" passes
    assert ce.passed_at_review(conf, ids, 0.0).all()             # review nothing: all pass


def ci(lo, hi):
    """A CI for a - b, and the matching one for b - a."""
    return (lo, hi), (-hi, -lo)


def diff_table(pe_vs_cur, wang_vs_cur, wang_vs_pe):
    """Every ordered pair's AURC-difference CI, as decide() expects them."""
    d = {}
    d[("s_pe", "s_current")], d[("s_current", "s_pe")] = ci(*pe_vs_cur)
    d[("s_wang", "s_current")], d[("s_current", "s_wang")] = ci(*wang_vs_cur)
    d[("s_wang", "s_pe")], d[("s_pe", "s_wang")] = ci(*wang_vs_pe)
    return d


GOOD = {"s_current": 0.001, "s_pe": 0.001, "s_wang": 0.001}  # every score meets WRC's 1.5% bar


def test_a_tied_reference_score_replaces_ours():
    aurc = {"s_current": 0.010, "s_pe": 0.012, "s_wang": 0.011}
    d = diff_table(pe_vs_cur=(-0.001, 0.005), wang_vs_cur=(-0.002, 0.004), wang_vs_pe=(-0.003, 0.001))  # all CIs hold 0
    out = ce.decide(aurc, d, GOOD)
    assert out["lowest_aurc"] == "s_current" and set(out["tied_set"]) == {"s_current", "s_pe", "s_wang"}
    assert out["chosen"] == "s_pe" and out["outcome"] == "A"  # [ref] first, then the fewest passes


def test_ours_is_kept_only_when_no_reference_score_ties_it():
    aurc = {"s_current": 0.010, "s_pe": 0.020, "s_wang": 0.015}
    d = diff_table(pe_vs_cur=(0.005, 0.015), wang_vs_cur=(0.001, 0.009), wang_vs_pe=(-0.010, 0.000))  # both worse
    out = ce.decide(aurc, d, GOOD)
    assert out["chosen"] == "s_current" and out["outcome"] == "B"   # what happened with the real data


def test_a_better_reference_score_wins_and_failing_the_wrc_bar_means_c():
    aurc = {"s_current": 0.012, "s_pe": 0.020, "s_wang": 0.009}
    d = diff_table(pe_vs_cur=(0.004, 0.012), wang_vs_cur=(-0.005, -0.001), wang_vs_pe=(-0.015, -0.006))
    assert ce.decide(aurc, d, GOOD)["chosen"] == "s_wang"            # clearly lowest
    bad = {**GOOD, "s_wang": 0.02}                                  # ... but it passes 2% errors
    out = ce.decide(aurc, d, bad)
    assert out["chosen"] == "s_wang" and not out["q1_passes"] and out["outcome"] == "C"


def test_error_among_passed_uses_the_matched_review_set():
    d = pd.DataFrame({"scan_id": list("abcd"), "s": [0.9, 0.8, 0.7, 0.1], "error": [False, True, False, True]})
    assert ce.error_among_passed(d, "s", 0.25) == pytest.approx(1 / 3)  # d reviewed; b is a passed error
