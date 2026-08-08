"""Needs scipy, which the trainer's dependencies do not include.

Without the skip this fails at *collection*, which pytest reports as an error rather than a skip
and turns the whole run red even though nothing here is broken.
"""

import pytest

pytest.importorskip("scipy")

import numpy as np
from projects.student_state_tutor import sphere_curriculum_value


def test_row_pick_selects_per_student_domain():
    values = np.asarray([[0.1, 0.9], [0.8, 0.2]])
    selected = sphere_curriculum_value.row_pick(values, np.asarray([0, 1]))
    np.testing.assert_allclose(selected, [0.1, 0.2])


def test_rank_correlation_detects_matching_and_reversed_order():
    left = np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    right = np.asarray([[0.2, 0.4, 0.6], [0.6, 0.4, 0.2]])
    correlations = sphere_curriculum_value.rank_correlation(left, right)
    np.testing.assert_allclose(correlations, [1.0, -1.0])
