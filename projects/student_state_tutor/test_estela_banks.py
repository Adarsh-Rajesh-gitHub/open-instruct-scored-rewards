import json

import yaml
from projects.student_state_tutor import estela_banks


def question(question_id="q-1", image=False):
    payload = {
        "id": question_id,
        "text": "Which interaction is a third-law pair?",
        "answers": [
            {"answer": {"text": "A and B", "correct": True}},
            {"answer": {"text": "A and C", "correct": False}},
            {"answer": {"text": "B and C", "correct": False}},
        ],
    }
    if image:
        payload["image"] = "diagram.png"
    return {"multiple_choice": payload}


def test_parse_question_accepts_one_correct_text_item():
    parsed, reason = estela_banks.parse_question(question(), "bank", 1)
    assert reason == "valid"
    assert parsed["gold_idx"] == 0
    assert len(parsed["choices"]) == 3


def test_parse_question_rejects_image_reference():
    parsed, reason = estela_banks.parse_question(question(image=True), "bank", 1)
    assert parsed is None
    assert reason == "image_reference"


def test_catalog_parser_supports_numerical_answer_keys():
    parsed, reason = estela_banks.parse_catalog_question(
        {
            "numerical": {
                "id": "n-1",
                "text": "What is the acceleration?",
                "answer": {"value": 9.8, "margin_type": "percent", "tolerance": 3},
            }
        },
        "bank",
        1,
    )
    assert reason == "valid"
    assert parsed["question_type"] == "numerical"
    assert parsed["answer_key"]["value"] == 9.8


def test_extract_banks_keeps_ready_canonical_version(tmp_path):
    unit = tmp_path / "PHY I Mechanics" / "3_Forces" / "BANK"
    unit.mkdir(parents=True)
    document = {
        "bank_info": {"bank_id": "PHY1-F-TEST", "title": "Test bank", "status": "ready"},
        "questions": [question(f"q-{index}") for index in range(6)],
    }
    (unit / "bank.yaml").write_text(yaml.safe_dump(document))
    old = unit / "Old"
    old.mkdir()
    (old / "bank.yaml").write_text(yaml.safe_dump(document))

    banks, stats = estela_banks.extract_banks(tmp_path, min_questions=6)

    assert len(banks) == 1
    assert banks[0]["concept_id"] == "phy1_f_test"
    assert banks[0]["unit"] == "Forces"
    assert stats["selected_questions"] == 6
    assert stats["filter_reasons"]["archive_path"] == 1


def test_extract_catalog_uses_canonical_root_bank_files(tmp_path):
    unit = tmp_path / "PHY I Mechanics" / "3_Forces" / "BANK"
    unit.mkdir(parents=True)
    document = {
        "bank_info": {"bank_id": "metadata-id", "title": "Test bank", "status": "ready"},
        "questions": [question("q-1"), question("q-2")],
    }
    (unit / "BANK.yaml").write_text(yaml.safe_dump(document))
    old = unit / "Old"
    old.mkdir()
    (old / "BANK.yaml").write_text(yaml.safe_dump(document))

    banks, stats = estela_banks.extract_catalog(tmp_path, min_questions=2)

    assert len(banks) == 1
    assert banks[0]["bank_id"] == "BANK"
    assert stats["selected_questions"] == 2


def test_write_jsonl_round_trip(tmp_path):
    out = tmp_path / "banks.jsonl"
    estela_banks.write_jsonl(out, [{"bank_id": "one"}])
    assert json.loads(out.read_text()) == {"bank_id": "one"}
