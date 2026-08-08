"""Extract answer-keyed, text-only isomorphic banks from ESTELA YAML files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ARCHIVE_PARTS = {
    "archive",
    "draft",
    "drafts",
    "files",
    "old",
    "old files",
    "older versions",
    "scripts",
    "templates",
    "version with alternate numbers",
}
IMAGE_PATTERN = re.compile(r"(?:<img\b|!\[[^\]]*\]\(|\.(?:png|jpe?g|gif|svg|webp)\b)", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = HTML_PATTERN.sub(" ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def has_image_reference(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if any(token in str(key).lower() for token in ("image", "figure")):
                return True
            if has_image_reference(child):
                return True
        return False
    if isinstance(value, list):
        return any(has_image_reference(child) for child in value)
    return bool(IMAGE_PATTERN.search(str(value)))


def unwrap_question(row: dict) -> tuple[str, dict] | None:
    supported = {"multiple_choice", "multiple-choice", "multiple choice"}
    for question_type, payload in row.items():
        if question_type.strip().lower() in supported and isinstance(payload, dict):
            return "multiple_choice", payload
    return None


def parse_answers(payload: dict) -> tuple[list[str], int] | None:
    choices = []
    correct_indices = []
    for row in payload.get("answers", []):
        answer = row.get("answer", row) if isinstance(row, dict) else None
        if not isinstance(answer, dict):
            return None
        text = clean_text(answer.get("text"))
        if not text:
            return None
        choices.append(text)
        if is_true(answer.get("correct", False)):
            correct_indices.append(len(choices) - 1)
    if len(choices) < 3 or len(correct_indices) != 1:
        return None
    return choices, correct_indices[0]


def parse_catalog_answer(question_type: str, payload: dict) -> dict | None:
    if question_type == "multiple_choice":
        parsed = parse_answers(payload)
        if parsed is None:
            return None
        choices, gold_idx = parsed
        return {"choices": choices, "gold_idx": gold_idx}
    if question_type == "numerical":
        answer = payload.get("answer")
        if not isinstance(answer, dict):
            return None
        if answer.get("value") is None and not (
            answer.get("range_start") is not None and answer.get("range_end") is not None
        ):
            return None
        fields = ("value", "range_start", "range_end", "margin_type", "tolerance", "precision_type", "precision")
        return {key: answer[key] for key in fields if answer.get(key) is not None}
    if question_type == "multiple_answers":
        choices = []
        correct_indices = []
        for row in payload.get("answers", []):
            answer = row.get("answer", row) if isinstance(row, dict) else None
            if not isinstance(answer, dict):
                return None
            text = clean_text(answer.get("text"))
            if not text:
                return None
            choices.append(text)
            if is_true(answer.get("correct", False)):
                correct_indices.append(len(choices) - 1)
        if len(choices) < 2 or not correct_indices:
            return None
        return {"choices": choices, "correct_indices": correct_indices}
    if question_type == "categorization":
        categories = []
        for row in payload.get("categories", []):
            category = row.get("category", row) if isinstance(row, dict) else None
            if not isinstance(category, dict):
                return None
            description = clean_text(category.get("description"))
            answers = [clean_text(answer) for answer in (category.get("answers") or []) if clean_text(answer)]
            if not description:
                return None
            categories.append({"description": description, "answers": answers})
        if not categories or not any(category["answers"] for category in categories):
            return None
        return {
            "categories": categories,
            "distractors": [clean_text(value) for value in payload.get("distractors", []) if clean_text(value)],
        }
    return None


def parse_catalog_question(row: dict, bank_id: str, index: int) -> tuple[dict | None, str]:
    if len(row) != 1:
        return None, "invalid_question_wrapper"
    question_type, payload = next(iter(row.items()))
    if question_type not in {"numerical", "multiple_choice", "multiple_answers", "categorization"} or not isinstance(
        payload, dict
    ):
        return None, "unsupported_type"
    if has_image_reference(payload):
        return None, "image_reference"
    text = clean_text(payload.get("text"))
    if not text:
        return None, "missing_text"
    answer_key = parse_catalog_answer(question_type, payload)
    if answer_key is None:
        return None, "invalid_answers"
    feedback = payload.get("feedback", {})
    question = {
        "question_id": clean_text(payload.get("id")) or f"{bank_id}_q{index:03d}",
        "title": clean_text(payload.get("title")),
        "question": text,
        "question_type": question_type,
        "answer_key": answer_key,
        "feedback": (clean_text(feedback.get("general")) if isinstance(feedback, dict) else ""),
    }
    if question_type == "multiple_choice":
        question.update(answer_key)
    return question, "valid"


def parse_question(row: dict, bank_id: str, index: int) -> tuple[dict | None, str]:
    unwrapped = unwrap_question(row)
    if unwrapped is None:
        return None, "unsupported_type"
    question_type, payload = unwrapped
    if has_image_reference(payload):
        return None, "image_reference"
    text = clean_text(payload.get("text"))
    if not text:
        return None, "missing_text"
    parsed_answers = parse_answers(payload)
    if parsed_answers is None:
        return None, "invalid_answers"
    choices, gold_idx = parsed_answers
    feedback = payload.get("feedback", {})
    general_feedback = clean_text(feedback.get("general")) if isinstance(feedback, dict) else ""
    return (
        {
            "question_id": clean_text(payload.get("id")) or f"{bank_id}_q{index:03d}",
            "title": clean_text(payload.get("title")),
            "question": text,
            "choices": choices,
            "gold_idx": gold_idx,
            "feedback": general_feedback,
            "question_type": question_type,
        },
        "valid",
    )


def infer_unit(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    for part in relative.parts:
        if re.match(r"^\d+_", part):
            return re.sub(r"^\d+_", "", part).replace("_", " ")
    return relative.parent.name.replace("_", " ")


def parse_bank(path: Path, root: Path) -> tuple[dict | None, Counter]:
    reasons = Counter()
    try:
        document = yaml.safe_load(path.read_text())
    except Exception:
        reasons["yaml_error"] += 1
        return None, reasons
    if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
        reasons["invalid_document"] += 1
        return None, reasons

    info = document.get("bank_info", {})
    if not isinstance(info, dict):
        info = {}
    bank_id = clean_text(info.get("bank_id")) or path.stem
    questions = []
    for index, row in enumerate(document["questions"], start=1):
        if not isinstance(row, dict):
            reasons["invalid_question"] += 1
            continue
        question, reason = parse_question(row, bank_id, index)
        reasons[reason] += 1
        if question is not None:
            questions.append(question)

    return (
        {
            "bank_id": bank_id,
            "title": clean_text(info.get("title")) or path.parent.name,
            "description": clean_text(info.get("description")),
            "status": clean_text(info.get("status")).lower(),
            "unit": infer_unit(path, root),
            "source_path": str(path.relative_to(root)),
            "questions": questions,
        },
        reasons,
    )


def is_archive_path(path: Path, root: Path) -> bool:
    parts = {part.strip().lower() for part in path.relative_to(root).parts[:-1]}
    return bool(parts & ARCHIVE_PARTS)


def extract_banks(root: Path, min_questions: int = 6) -> tuple[list[dict], dict]:
    candidates: dict[str, list[dict]] = {}
    reasons = Counter()
    scanned = 0
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        if is_archive_path(path, root):
            reasons["archive_path"] += 1
            continue
        scanned += 1
        bank, bank_reasons = parse_bank(path, root)
        reasons.update(bank_reasons)
        if bank is None:
            continue
        if bank["status"] and bank["status"] != "ready":
            reasons["not_ready"] += 1
            continue
        candidates.setdefault(bank["bank_id"], []).append(bank)

    selected = []
    for bank_id, versions in candidates.items():
        best = max(
            versions,
            key=lambda bank: (
                len(bank["questions"]),
                bank["status"] == "ready",
                -len(Path(bank["source_path"]).parts),
            ),
        )
        if len(best["questions"]) < min_questions:
            reasons["too_few_valid_questions"] += 1
            continue
        best["concept_id"] = re.sub(r"[^a-z0-9]+", "_", bank_id.lower()).strip("_")
        selected.append(best)
    selected.sort(key=lambda bank: (bank["unit"], bank["bank_id"]))

    stats = {
        "yaml_files_scanned": scanned,
        "unique_bank_ids": len(candidates),
        "selected_banks": len(selected),
        "selected_questions": sum(len(bank["questions"]) for bank in selected),
        "units": dict(Counter(bank["unit"] for bank in selected)),
        "filter_reasons": dict(reasons),
    }
    return selected, stats


def canonical_bank_paths(root: Path) -> list[Path]:
    grouped: dict[Path, list[Path]] = {}
    for path in sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")]):
        parts = path.relative_to(root).parts
        if len(parts) != 4 or parts[0] != "PHY I Mechanics":
            continue
        grouped.setdefault(path.parent, []).append(path)
    selected = []
    for folder, paths in grouped.items():
        preferred = [path for path in paths if path.stem.lower() == folder.name.lower()]
        selected.append(sorted(preferred or paths)[0])
    return sorted(selected)


def extract_catalog(root: Path, min_questions: int = 2) -> tuple[list[dict], dict]:
    selected = []
    reasons = Counter()
    paths = canonical_bank_paths(root)
    for path in paths:
        try:
            document = yaml.safe_load(path.read_text())
        except Exception:
            reasons["yaml_error"] += 1
            continue
        if not isinstance(document, dict) or not isinstance(document.get("questions"), list):
            reasons["invalid_document"] += 1
            continue
        info = document.get("bank_info", {})
        if not isinstance(info, dict):
            info = {}
        status = clean_text(info.get("status")).lower()
        if status not in {"ready", "deployed"}:
            reasons["not_ready"] += 1
            continue
        bank_id = path.parent.name
        questions = []
        bank_reasons = Counter()
        for index, row in enumerate(document["questions"], start=1):
            if not isinstance(row, dict):
                bank_reasons["invalid_question"] += 1
                continue
            question, reason = parse_catalog_question(row, bank_id, index)
            bank_reasons[reason] += 1
            if question is not None:
                questions.append(question)
        reasons.update(bank_reasons)
        if bank_reasons["image_reference"]:
            reasons["image_bank"] += 1
            continue
        if len(questions) != len(document["questions"]):
            reasons["partially_invalid_bank"] += 1
            continue
        if len(questions) < min_questions:
            reasons["too_few_valid_questions"] += 1
            continue
        selected.append(
            {
                "bank_id": bank_id,
                "concept_id": re.sub(r"[^a-z0-9]+", "_", bank_id.lower()).strip("_"),
                "title": clean_text(info.get("title")) or path.parent.name,
                "description": clean_text(info.get("description")),
                "status": status,
                "unit": infer_unit(path, root),
                "source_path": str(path.relative_to(root)),
                "questions": questions,
            }
        )
    stats = {
        "canonical_bank_files": len(paths),
        "selected_banks": len(selected),
        "selected_questions": sum(len(bank["questions"]) for bank in selected),
        "question_types": dict(
            Counter(question["question_type"] for bank in selected for question in bank["questions"])
        ),
        "units": dict(Counter(bank["unit"] for bank in selected)),
        "filter_reasons": dict(reasons),
    }
    return selected, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estela-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path)
    parser.add_argument("--min-questions", type=int, default=6)
    parser.add_argument("--mode", choices=("choice_scoring", "catalog"), default="choice_scoring")
    args = parser.parse_args()

    if args.mode == "catalog":
        banks, stats = extract_catalog(args.estela_root, args.min_questions)
    else:
        banks, stats = extract_banks(args.estela_root, args.min_questions)
    write_jsonl(args.out, banks)
    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
