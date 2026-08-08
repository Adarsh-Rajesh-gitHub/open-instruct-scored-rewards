"""Needs scipy, which the trainer's dependencies do not include.

Without the skip this fails at *collection*, which pytest reports as an error rather than a skip
and turns the whole run red even though nothing here is broken.
"""

import pytest

pytest.importorskip("scipy")

import numpy as np
from projects.student_state_tutor import sphere_state_update_curve


def test_split_with_k_history_uses_exact_count_per_domain():
    domains = np.repeat(np.arange(3), [6, 7, 8])
    history, target = sphere_state_update_curve.split_with_k_history(domains, k=2, rng=np.random.default_rng(0))
    assert [int(np.sum(domains[history] == domain)) for domain in range(3)] == [2, 2, 2]
    assert len(history) + len(target) == len(domains)
    assert not set(history) & set(target)
