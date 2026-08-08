import json

from projects.student_state_tutor import memory_state_ablation


def test_dialogue_turns_parses_multiturn_completion():
    completion = (
        "Student: I think current gets used up.\n"
        "Tutor: What happens before and after the bulb?\n"
        "Student: Maybe less comes out afterward?"
    )
    assert memory_state_ablation.dialogue_turns(completion) == [
        ("Student", "I think current gets used up."),
        ("Tutor", "What happens before and after the bulb?"),
        ("Student", "Maybe less comes out afterward?"),
    ]


def test_load_memory_examples_keeps_longest_unresolved_trace(tmp_path):
    graph_row = {
        "question": "What happens to current?",
        "subject": "science",
        "graph_actual": {
            "concept": "electric current",
            "mistaken_relation": "current is consumed",
        },
    }
    graph_path = tmp_path / "graph.jsonl"
    graph_path.write_text(json.dumps(graph_row) + "\n")

    short = {
        "prompt": graph_row["question"],
        "solved": 0.0,
        "completion": "Student: Unsure.\nTutor: Think.\nStudent: Still unsure.",
    }
    long = {
        "prompt": graph_row["question"],
        "solved": 0.0,
        "completion": (
            "Student: Unsure.\nTutor: Think.\nStudent: Maybe used up.\n"
            "Tutor: Compare both sides.\nStudent: I still think less comes out."
        ),
    }
    traces_path = tmp_path / "traces.jsonl"
    traces_path.write_text(json.dumps(short) + "\n" + json.dumps(long) + "\n")

    examples = memory_state_ablation.load_memory_examples(
        traces_path, graph_path, limit=10
    )

    assert len(examples) == 1
    assert examples[0]["history_student_turns"] == 3
    assert examples[0]["last_student_turn"] == "I still think less comes out."
