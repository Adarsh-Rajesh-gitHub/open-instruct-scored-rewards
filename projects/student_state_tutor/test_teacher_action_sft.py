from collections import Counter

from projects.student_state_tutor import teacher_action_sft


def bank(index):
    concept_id = f"skill_{index}"
    return {
        "bank_id": f"BANK-{index}",
        "concept_id": concept_id,
        "title": f"Skill {index}",
        "questions": [{"question_id": "q-1", "question": "Question", "choices": ["A", "B", "C"], "gold_idx": 0}],
    }


def test_sft_examples_balance_actions_and_targets():
    examples = teacher_action_sft.build_sft_examples(
        [bank(index) for index in range(4)], examples_per_action=8, seed=0
    )
    action_counts = Counter(row["decision"]["action"] for row in examples)
    target_counts = Counter(row["decision"]["target_node"] for row in examples)
    assert action_counts == {
        "diagnostic": 8,
        "worked_example": 8,
        "contrast_case": 8,
        "retrieval_practice": 8,
        "stop": 8,
    }
    assert set(target_counts) == {f"skill_{index}" for index in range(4)}
    for row in examples:
        decision = row["decision"]
        learner_id = row["case"]["learner_id"]
        assert decision["action"] not in learner_id
        assert decision["stop"] == (decision["action"] == "stop")
        if decision["action"] == "stop":
            assert decision["item_id"] is None
