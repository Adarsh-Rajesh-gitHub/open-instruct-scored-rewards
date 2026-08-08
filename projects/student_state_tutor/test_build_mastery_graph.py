import json

from projects.student_state_tutor import build_mastery_graph


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_join_and_build_mastery_graph(tmp_path):
    items_path = tmp_path / "items.jsonl"
    annotations_path = tmp_path / "units.jsonl"
    write_jsonl(
        items_path,
        [
            {
                "question": "Why does day become night?",
                "choices": ["rotation", "clouds"],
                "gold_idx": 0,
                "subject": "science",
                "grade": 6,
            },
            {
                "question": "Math item",
                "choices": ["1", "2"],
                "gold_idx": 0,
                "subject": "math",
            },
        ],
    )
    write_jsonl(
        annotations_path,
        [
            {
                "key": "why does day become night?",
                "units": ["Earth rotation"],
                "prerequisites": [["Earth rotation", "cyclic motion"]],
                "misconception": "Attributes day and night to clouds.",
            }
        ],
    )

    rows, unmatched = build_mastery_graph.join_items(
        items_path, annotations_path, "train"
    )
    graph, summary, pairs = build_mastery_graph.build(
        rows, {"train": unmatched, "eval": 0}
    )

    assert len(rows) == 1
    assert summary["nodes"] == 2
    assert summary["edges"] == 1
    assert summary["items"]["train"] == 1
    assert not pairs
    assert {node["id"] for node in graph["nodes"]} == {
        "earth rotation",
        "cyclic motion",
    }


def test_exact_transfer_pairs_cross_split():
    rows = [
        {"key": "a", "split": "train", "units": ["waves"]},
        {"key": "b", "split": "eval", "units": ["waves"]},
        {"key": "c", "split": "eval", "units": ["cells"]},
    ]
    assert build_mastery_graph.exact_transfer_pairs(rows, "train", "eval") == [
        {"unit": "waves", "left": "a", "right": "b"}
    ]
