"""Phase C's two pieces of new logic: the vectorised cutoff must equal gate_analysis's, and the
reading schedule must follow the gear's order (visits by rank; inside a visit, flagged first with
no-retina failures first, then least confident; then the rest, least confident first)."""
import numpy as np   # random test data
import pandas as pd  # tables

from analysis import gate_analysis as ga   # the original cutoff rule
from analysis import phase_c                # the code under test


def test_fast_youden_equals_the_original():
    rng = np.random.default_rng(0)
    for _ in range(50):                                             # many random draws, with ties
        conf = np.round(rng.uniform(0.9, 1.0, 200), 3)
        ok = rng.random(200) < 0.9
        assert phase_c.fast_youden(conf, ok) == ga.youden_threshold(conf, ok)
    assert phase_c.fast_youden(np.array([0.5, 0.6]), np.array([True, True])) == 0.5  # one class: pass all


def test_schedule_reads_whole_visits_in_rank_order_and_the_gears_order_inside():
    b = pd.DataFrame({
        "patient":      ["V2", "V2", "V2", "V1", "V1", "V1"],
        "bscan":        [1, 2, 3, 1, 2, 3],
        "confidence":   [0.99, 0.95, 0.97, 0.90, 0.99, 0.98],
        "review_final": [False, True, True, True, False, False],   # the gate's flags
        "retina_ok":    [True, True, False, True, True, True],     # V2 B-scan 3: no retina, a hard failure
        "error":        [False, True, False, True, False, False]})
    v = pd.DataFrame({"visit": ["V1", "V2"], "rank": [2, 1], "tier": [2, 1]})
    s = phase_c.schedule(b, v)
    assert list(zip(s.patient, s.bscan)) == [("V2", 3), ("V2", 2), ("V2", 1),   # V2 first (rank 1): hard failure,
                                             ("V1", 1), ("V1", 3), ("V1", 2)]   # then least confident, then the rest
    assert s.poor_found.tolist() == [0, 1, 1, 2, 2, 2] and s.read.tolist() == [1, 2, 3, 4, 5, 6]
