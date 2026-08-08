from projects.student_state_tutor import teacher_action_ablation
from projects.student_state_tutor import teacher_transfer_ablation


def bank(index):
    concept_id = f"skill_{index}"
    return {
        "bank_id": f"BANK-{index}",
        "concept_id": concept_id,
        "title": f"Skill {index}",
        "description": f"Description {index}",
        "questions": [
            {
                "question_id": f"q-{question_index}",
                "question": f"Question {question_index}",
                "choices": ["A", "B", "C"],
                "gold_idx": 0,
                "feedback": "Use the governing principle.",
            }
            for question_index in range(4)
        ],
    }


def screened_questions(banks):
    rows = teacher_transfer_ablation.flatten_questions(banks)
    for row in rows:
        row["baseline"] = {
            "scores": [-2.0, -1.0, -3.0],
            "predicted_idx": 1,
            "gold_probability": 0.2,
            "solved": 0,
        }
    return rows


def test_build_transfer_rows_requires_distinct_wrong_items():
    banks = [bank(index) for index in range(4)]
    action_rows = teacher_action_ablation.build_cases(
        banks, replicates=2, seed=0
    )
    for row in action_rows:
        row["decisions"] = {
            condition: {
                "valid": True,
                "target_node": row["oracle_target"],
                "action": row["oracle_action"],
                "item_id": "",
                "stop": False,
            }
            for condition in teacher_action_ablation.CONDITIONS
        }
    rows, stats = teacher_transfer_ablation.build_transfer_rows(
        banks,
        action_rows,
        screened_questions(banks),
        limit=8,
    )
    assert len(rows) == 8
    assert len(stats["eligible_concepts"]) == 4
    assert all(
        row["diagnostic"]["question_id"] != row["transfer"]["question_id"]
        for row in rows
    )


def test_canonical_intervention_uses_bank_feedback():
    banks = [bank(index) for index in range(4)]
    action_rows = teacher_action_ablation.build_cases(
        banks, replicates=1, seed=0
    )
    for row in action_rows:
        row["decisions"] = {}
    rows, _ = teacher_transfer_ablation.build_transfer_rows(
        banks,
        action_rows,
        screened_questions(banks),
        limit=1,
    )
    assert teacher_transfer_ablation.canonical_intervention(rows[0]) == (
        "Use the governing principle."
    )
