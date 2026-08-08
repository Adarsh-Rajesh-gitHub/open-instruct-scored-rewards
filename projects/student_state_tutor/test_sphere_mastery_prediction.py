import numpy as np

from projects.student_state_tutor import sphere_mastery_prediction


def test_posterior_mean_uses_beta_smoothing():
    observations = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.float32)
    means = sphere_mastery_prediction.posterior_mean(observations, axis=1)
    np.testing.assert_allclose(means, [3 / 5, 1 / 5])


def test_student_states_are_domain_specific():
    correctness = np.asarray(
        [
            [1, 1, 0, 0, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=np.float32,
    )
    domains = np.asarray([0, 0, 0, 1, 1, 1])
    history = np.asarray([0, 1, 3, 4])
    targets = np.asarray([2, 5])
    pseudo_domains = domains.copy()

    global_state, mastery, shuffled = sphere_mastery_prediction.student_states(
        correctness, domains, history, targets, pseudo_domains
    )

    np.testing.assert_allclose(global_state, [4 / 6, 2 / 6])
    np.testing.assert_allclose(mastery[0], [3 / 4, 2 / 4])
    np.testing.assert_allclose(mastery, shuffled)


def test_item_split_keeps_history_in_every_domain():
    domains = np.repeat(np.arange(3), [10, 8, 6])
    history, target = sphere_mastery_prediction.split_items(
        domains, np.random.default_rng(0)
    )
    assert set(domains[history]) == {0, 1, 2}
    assert set(domains[target]) == {0, 1, 2}
    assert not set(history) & set(target)
