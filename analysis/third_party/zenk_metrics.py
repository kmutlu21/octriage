# Risk-coverage metrics from the failure-detection benchmark of Zenk et al., "Comparative
# benchmarking of failure detection methods in medical image segmentation: Unveiling the role of
# confidence aggregation", Medical Image Analysis 2024 (doi:10.1016/j.media.2024.103392).
#
# Source: https://github.com/MIC-DKFZ/segmentation_failures_benchmark @ a1af98be0f93c2bdc30ffe5bb6eda531c485d87d
#         src/segmentation_failures/evaluation/failure_detection/metrics.py
#         (itself adapted from https://github.com/IML-DKFZ/fd-shifts), Apache License 2.0; the
#         licence text is in LICENSE-segmentation_failures_benchmark next to this file.
#
# MODIFIED (Apache-2.0 section 4b): kept only the risk-coverage metrics (StatsCache.rc_curve_stats,
# aurc, e-aurc, opt-aurc, rand-aurc, norm-aurc) and the registry; removed the ROC/AP/correlation
# metrics and their sklearn/scipy imports; ParamSpec imported from typing instead of
# typing_extensions. The kept code is otherwise unchanged.
# Adapted from https://github.com/IML-DKFZ/fd-shifts/tree/main
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, ParamSpec, TypeVar, cast

import numpy as np
import numpy.typing as npt

_metric_funcs = {}

T = TypeVar("T")
P = ParamSpec("P")


def may_raise_sklearn_exception(func: Callable[P, T]) -> Callable[P, T]:
    def _inner_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except ValueError:
            return cast(T, np.nan)

    return _inner_wrapper


def compute_optimal_aurc(risks):
    tmp_stat = StatsCache(confids=-risks, risks=risks)
    return aurc(tmp_stat)


def compute_random_aurc(risks: np.ndarray):
    return np.mean(risks)


@dataclass
class StatsCache:
    """Cache for stats computed by scikit used by multiple metrics.

    Attributes:
        confids (array_like): Confidence values associated with the predictions
        risks (array_like): Risk values associated with the predictions
    """

    confids: npt.NDArray[Any]
    risks: npt.NDArray[Any]

    @cached_property
    def rc_curve_stats(self) -> tuple[list[float], list[float], list[float]]:
        coverages = []
        selective_risks = []
        assert (
            len(self.risks.shape) == 1
            and len(self.confids.shape) == 1
            and len(self.risks) == len(self.confids)
        )

        n_samples = len(self.risks)
        idx_sorted = np.argsort(self.confids)

        coverage = n_samples
        error_sum = sum(self.risks[idx_sorted])

        coverages.append(coverage / n_samples)
        selective_risks.append(error_sum / n_samples)

        weights = []

        tmp_weight = 0
        for i in range(0, len(idx_sorted) - 1):
            coverage = coverage - 1
            error_sum = error_sum - self.risks[idx_sorted[i]]
            tmp_weight += 1
            if i == 0 or self.confids[idx_sorted[i]] != self.confids[idx_sorted[i - 1]]:
                coverages.append(coverage / n_samples)
                selective_risks.append(error_sum / (n_samples - 1 - i))
                weights.append(tmp_weight / n_samples)
                tmp_weight = 0

        # add a well-defined final point to the RC-curve.
        if tmp_weight > 0:
            coverages.append(0)
            selective_risks.append(selective_risks[-1])
            weights.append(tmp_weight / n_samples)

        return coverages, selective_risks, weights


def register_metric_func(name: str) -> Callable:
    def _inner_wrapper(func: Callable) -> Callable:
        _metric_funcs[name] = func
        return func

    return _inner_wrapper


def get_metric_function(metric_name: str) -> Callable[[StatsCache], float]:
    return _metric_funcs[metric_name]


@register_metric_func("aurc")
@may_raise_sklearn_exception
def aurc(stats_cache: StatsCache):
    _, risks, weights = stats_cache.rc_curve_stats
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])


@register_metric_func("e-aurc")
@may_raise_sklearn_exception
def eaurc(stats_cache: StatsCache):
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    aurc_opt = compute_optimal_aurc(stats_cache.risks)
    return aurc(stats_cache) - aurc_opt


@register_metric_func("opt-aurc")
@may_raise_sklearn_exception
def optimal_aurc(stats_cache: StatsCache):
    return compute_optimal_aurc(stats_cache.risks)


@register_metric_func("rand-aurc")
@may_raise_sklearn_exception
def random_aurc(stats_cache: StatsCache):
    return compute_random_aurc(stats_cache.risks)


@register_metric_func("norm-aurc")
@may_raise_sklearn_exception
def normalized_aurc(stats_cache: StatsCache):
    """Compute actually normalized AURC, i.e. normalize to range between optimal (1) and random CSF (0)."""
    aurc_opt = compute_optimal_aurc(stats_cache.risks)
    aurc_rand = compute_random_aurc(stats_cache.risks)
    return (aurc_rand - aurc(stats_cache)) / (aurc_rand - aurc_opt)
