"""The statistics behind every reported number: ICC, AUC, Spearman, the Youden cutoff,
the cross-validated gate (no fold chooses its own cutoff) and the patient-level bootstrap."""
import numpy as np       # arrays
import pandas as pd      # tables
import pytest            # the test framework

from analysis import gate_analysis as ga  # the code under test


def test_icc_matches_shrout_fleiss_worked_example():
    # Shrout & Fleiss (1979), 6 targets x 4 judges; they report ICC(2,1) = .29.
    x = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)
    assert round(ga.icc_a1(x), 2) == 0.29                                   # the published answer
    assert ga.icc_a1(np.repeat(np.arange(10.0)[:, None], 3, axis=1)) == pytest.approx(1.0)  # 3 identical raters


def test_auc_is_rank_based_with_ties_counted_half():
    assert ga.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0    # positives all score higher: perfect
    assert ga.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0    # all lower: perfectly wrong
    assert ga.auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5    # all tied: a coin flip
    assert np.isnan(ga.auc([0.5, 0.6], [1, 1]))                 # no negatives: undefined


def test_spearman_is_monotone_invariant():
    x = np.arange(20.0)
    assert ga.spearman(x, np.exp(x)) == pytest.approx(1.0)      # same order, very different values
    assert ga.spearman(x, -x**3) == pytest.approx(-1.0)         # reversed order


def test_youden_picks_the_separating_cutoff():
    conf = np.array([0.99, 0.97, 0.95, 0.80, 0.70])
    ok = np.array([True, True, True, False, False])             # the two least confident are errors
    assert ga.youden_threshold(conf, ok) == 0.95                # passes exactly the three good ones
    assert ga.youden_threshold(conf, np.ones(5, bool)) == 0.70  # no errors: pass everything


def test_cv_gate_never_uses_a_folds_own_scans_to_choose_its_threshold():
    # Fold 1's errors sit at high confidence; if fold 1 could see itself, its cutoff would move.
    df = pd.DataFrame({"fold": [0] * 4 + [1] * 4, "confidence": [0.9, 0.8, 0.3, 0.2, 0.95, 0.9, 0.85, 0.1],
                       "retina_present_fraction": 1.0})
    error = np.array([False, False, True, True, True, True, False, False])
    passed, thresholds = ga.cv_gate(df, error)
    assert thresholds[1] == ga.youden_threshold(df.confidence[:4], ~error[:4]) == 0.8  # chosen on fold 0 only
    assert passed[4:].tolist() == [True, True, True, False]     # fold 1 gated with that cutoff


def test_gate_also_applies_the_retina_check():
    df = pd.DataFrame({"fold": [0, 0, 1, 1], "confidence": [0.9, 0.1, 0.9, 0.9],
                       "retina_present_fraction": [1.0, 1.0, 1.0, 0.5]})
    passed, _ = ga.cv_gate(df, np.array([False, True, False, False]))
    assert passed[3] == False  # confident, but no retina


def test_operating_reproduces_the_papers_headline_numbers():
    # 10 scans: 7 pass, 3 reviewed; 2 errors, both among the passed ones
    df = pd.DataFrame({"passed": [True] * 7 + [False] * 3, "error": [False] * 6 + [True] * 2 + [False] * 2})
    assert ga.operating(df) == pytest.approx({"reviewed": 0.3, "error_rate_without_gate": 0.2,
                                              "error_rate_among_passed": 1 / 7, "errors_caught": 0.5})


def test_bootstrap_resamples_patients_not_scans():
    # One patient with many identical scans must not shrink the interval as if they were independent.
    df = pd.DataFrame({"patient": ["a"] * 50 + ["b", "c"], "v": [1.0] * 50 + [0.0, 0.0]})
    est, lo, hi = ga.bootstrap(df, lambda d: d.v.mean(), n=500)
    assert est == pytest.approx(50 / 52)                        # the estimate uses every scan
    assert lo == 0.0 and hi > 0.95  # scan-level resampling would give roughly [0.90, 1.00]
