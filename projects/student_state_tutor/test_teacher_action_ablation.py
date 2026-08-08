import json

from projects.student_state_tutor import teacher_action_ablation


def bank(index):
    concept_id = f"skill_{index}"
    return {
        "bank_id": f"BANK-{index}",
        "concept_id": concept_id,
        "title": f"Skill {index}",
        "questions": [{"question_id": "q-1", "question": "Question", "choices": ["A", "B", "C"], "gold_idx": 0}],
    }


def test_build_cases_has_known_weakest_skill():
    cases = teacher_action_ablation.build_cases([bank(index) for index in range(4)], replicates=2, seed=0)
    assert len(cases) == 8
    for case in cases:
        node = next(node for node in case["graph_view"]["nodes"] if node["concept_id"] == case["oracle_target"])
        assert node["mean"] == min(row["mean"] for row in case["graph_view"]["nodes"])
        assert case["oracle_action"] in teacher_action_ablation.ACTIONS


def test_parse_decision_validates_target_item_pair():
    case = teacher_action_ablation.build_cases([bank(index) for index in range(4)], replicates=1, seed=0)[0]
    target = case["oracle_target"]
    item_id = next(row["item_id"] for row in case["candidates"] if row["concept_id"] == target)
    text = json.dumps({"target_node": target, "action": case["oracle_action"], "item_id": item_id, "stop": False})
    assert teacher_action_ablation.parse_decision(text, case)["valid"]

    wrong_item = next(row["item_id"] for row in case["candidates"] if row["concept_id"] != target)
    invalid = json.dumps({"target_node": target, "action": "contrast_case", "item_id": wrong_item, "stop": False})
    assert not teacher_action_ablation.parse_decision(invalid, case)["valid"]


def test_summary_passes_for_oracle_decisions():
    cases = teacher_action_ablation.build_cases([bank(index) for index in range(4)], replicates=2, seed=0)
    for case in cases:
        decisions = {}
        for condition in teacher_action_ablation.CONDITIONS:
            target = case["shuffled_visible_target"] if condition == "shuffled_graph" else case["oracle_target"]
            item_id = next(row["item_id"] for row in case["candidates"] if row["concept_id"] == target)
            decisions[condition] = {
                "valid": True,
                "target_node": target,
                "action": case["oracle_action"],
                "item_id": item_id,
                "stop": False,
            }
        case["decisions"] = decisions
    summary = teacher_action_ablation.summarize(cases)
    assert summary["conditions"]["graph"]["target_accuracy"] == 1.0
    assert summary["shuffled_visible_target_accuracy"] == 1.0
