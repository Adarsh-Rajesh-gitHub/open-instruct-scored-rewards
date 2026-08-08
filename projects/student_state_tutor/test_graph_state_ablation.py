from projects.student_state_tutor import graph_state_ablation


def test_parse_graph_state():
    parsed = graph_state_ablation.parse_graph_state(
        "CONCEPT: electric current\nMISTAKEN_RELATION: current is gradually consumed by components"
    )
    assert parsed is not None
    assert parsed["concept"] == "electric current"
    assert parsed["mistaken_relation"] == "current is gradually consumed by components"


def test_parse_graph_state_rejects_missing_relation():
    assert graph_state_ablation.parse_graph_state("CONCEPT: seasons") is None


def test_option_overlap_detection():
    row = {
        "choices": [
            "Earth spinning on its axis",
            "clouds moving over Earth",
            "the Moon blocking sunlight",
            "Earth orbiting the Sun",
        ],
        "gold_idx": 0,
        "belief_idx": 3,
        "counterfactual_idx": 2,
    }
    safe = {"concept": "day and night", "mistaken_relation": "daily light cycles result from annual revolution"}
    leaking = {"concept": "day and night", "mistaken_relation": "Earth orbiting the Sun"}
    assert not graph_state_ablation.has_option_overlap(safe, row)
    assert graph_state_ablation.has_option_overlap(leaking, row)
