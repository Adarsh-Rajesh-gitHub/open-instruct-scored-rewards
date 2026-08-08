import json

from projects.student_state_tutor import oracle_state_ablation


def test_extract_student_opening():
    completion = "Student: I thought current gets used up.\nTutor: Let's inspect that idea."
    assert oracle_state_ablation.extract_student_opening(completion) == "I thought current gets used up."


def test_load_examples_filters_math_and_deduplicates_questions(tmp_path):
    science = {
        "prompt": "What happens to current in a series circuit?",
        "completion": "Student: I think the bulb uses all the current.\nTutor: Why?",
        "choices": ["It is consumed", "It stays equal", "It doubles", "It stops"],
        "gold_idx": 1,
        "student_believed": "It is consumed",
        "subject": "science",
    }
    math = {**science, "prompt": "What is 2 + 2?", "subject": "math"}
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (science, science, math)) + "\n")

    examples = oracle_state_ablation.load_examples(path, limit=10, seed=0)

    assert len(examples) == 1
    assert examples[0]["belief_idx"] == 0
    assert examples[0]["counterfactual_idx"] in (2, 3)


def test_answer_leak_detection():
    example = {"choices": ["clouds", "Earth spinning on its axis", "the Moon", "wind"], "gold_idx": 1}
    assert oracle_state_ablation.leaks_answer("The answer is B.", example)
    assert oracle_state_ablation.leaks_answer("Think about Earth spinning on its axis.", example)
    assert not oracle_state_ablation.leaks_answer(
        "What daily motion could produce alternating light and darkness?", example
    )
