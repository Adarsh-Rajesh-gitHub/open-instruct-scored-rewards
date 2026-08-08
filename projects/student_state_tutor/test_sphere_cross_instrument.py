import numpy as np

from projects.student_state_tutor import sphere_cross_instrument


def test_build_features_repeats_student_state_per_target():
    features = sphere_cross_instrument.build_features(
        student_indices=np.asarray([1, 0]),
        target_items=np.asarray([2, 3]),
        item_difficulty=np.asarray([0.2, 0.8]),
        global_state=np.asarray([0.3, 0.7]),
        related_state=np.asarray([0.4, 0.9]),
        unrelated_state=np.asarray([0.5, 0.6]),
        shuffled_state=np.asarray([0.9, 0.4]),
    )
    np.testing.assert_allclose(
        features["related"],
        [
            [0.2, 0.7, 0.9],
            [0.8, 0.7, 0.9],
            [0.2, 0.3, 0.4],
            [0.8, 0.3, 0.4],
        ],
    )
    assert features["global"].shape == (4, 2)
