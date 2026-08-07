"""Stage 1 data-prep for the PRM-vs-ORM experiment.

Turns the raw PRM800K corpus (registered published dataset ``vendor/openai-prm800k``,
four ``.jsonl`` files under ``s3://edullm-data/vendor/openai-prm800k/v1/``) into the
corpora every later stage consumes, plus GSM8K/MATH prompt and eval sets:

  * ``sft.jsonl``           -- open-instruct ``messages`` SFT format, one chosen
                              trajectory per problem, completion-only loss (automatic).
  * ``orm.jsonl``           -- whole-solution outcome labels (1 correct / 0 incorrect),
                              graded by open-instruct's own MathVerifier.
  * ``prm.jsonl``           -- per-step examples ``(problem, prefix_steps, step, rating)``
                              with rating in {-1,0,+1}, taken up to the first negative
                              step of the chosen trajectory, plus rated alternatives.
  * ``prompts_math.jsonl``  -- MATH (PRM800K) train problems for GRPO, with answer.
  * ``prompts_gsm8k.jsonl`` -- GSM8K train problems for GRPO, with answer.
  * ``eval_math.jsonl``     -- MATH held-out (PRM800K *_test) problems for eval.
  * ``eval_gsm8k.jsonl``    -- GSM8K test problems for eval (primary metric).
  * ``MANIFEST.json``       -- row counts and the exact s3:// URIs written.

Runs as an eduLLM platform ``-train`` CPU job:

    python projects/prm_vs_orm/data_prep.py --out "$EDULLM_CHECKPOINT_DIR"

``$EDULLM_CHECKPOINT_DIR`` is an ``s3://`` prefix; the platform does NOT sync local
files, so every output is uploaded to S3 by this script using the job role's
credentials (via edullm_data's boto3 client). Reading PRM800K uses the same role;
neither path runs on a laptop.

``--selftest`` exercises the pure transform functions on a synthetic PRM800K record
with no S3, no network and no torch, so the logic can be linted and checked before a
submission (the same discipline as the milestone-1 plumbing smoke).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from typing import Any

#: Steps are separated by a blank line, both in the SFT target text and when a later
#: stage splits a completion back into steps to score. One separator, defined once.
STEP_SEP = "\n\n"


#: The answer surface form. MathVerifier extracts ``\boxed{}`` first and GSM8KVerifier
#: takes the last number in the text, so a solution ending in ``\boxed{<ans>}`` is
#: gradeable by both. (The plan's ``# Answer`` template is not what either grader reads.)
def _final_answer_line(answer: str) -> str:
    return f"The final answer is $\\boxed{{{answer}}}$."


def _has_boxed(text: str) -> bool:
    return "\\boxed{" in text


# --------------------------------------------------------------------------------------
# PRM800K record parsing (schema, per the openai/prm800k release):
#   record["question"]["problem"], record["question"]["ground_truth_answer"]
#   record["label"]["steps"][i]["completions"][j] = {"text", "rating" in {-1,0,1,None}}
#   record["label"]["steps"][i]["chosen_completion"] = index into completions, or None
#   record["label"]["steps"][i]["human_completion"]  = a human-written step, or None
#   phase-2 records additionally carry record["question"]["pre_generated_steps"].
# All accesses are defensive: fields may be null or absent across phases.
# --------------------------------------------------------------------------------------
def _problem(record: dict) -> str:
    q = record.get("question") or {}
    return (q.get("problem") or "").strip()


def _ground_truth(record: dict) -> str:
    q = record.get("question") or {}
    return (q.get("ground_truth_answer") or "").strip()


def _steps(record: dict) -> list[dict]:
    label = record.get("label") or {}
    return label.get("steps") or []


def chosen_trajectory(record: dict) -> list[dict]:
    """The steps actually taken, each as ``{"text", "rating"}``.

    Follows ``chosen_completion`` (or ``human_completion`` when no candidate was
    chosen) to the end of the labelled solution. A human-written step is treated as a
    positive (+1) step: it is the correction a human made because no candidate was good.
    """
    out: list[dict] = []
    for st in _steps(record):
        comps = st.get("completions") or []
        ci = st.get("chosen_completion")
        if ci is not None and 0 <= ci < len(comps):
            c = comps[ci] or {}
            out.append({"text": (c.get("text") or "").strip(), "rating": c.get("rating")})
            continue
        hc = st.get("human_completion")
        if hc:
            out.append({"text": str(hc).strip(), "rating": 1})
            continue
        # No chosen candidate and no human step -> the trajectory ends here.
        break
    return [s for s in out if s["text"]]


def solution_text(steps: Iterable[dict], answer: str) -> str:
    """Join step texts with the blank-line separator and guarantee a boxed answer."""
    parts = [s["text"] for s in steps if s.get("text")]
    body = STEP_SEP.join(parts)
    if not _has_boxed(body):
        body = (body + STEP_SEP if body else "") + _final_answer_line(answer)
    return body


def sft_record(problem: str, completion: str) -> dict:
    return {"messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": completion}]}


def prm_examples(record: dict) -> list[dict]:
    """Per-step training examples for the PRM.

    For each step position of the chosen trajectory (up to and including the first
    negative step), emit the chosen step and every rated alternative completion at that
    position, each sharing the prefix of chosen steps before it. The RM trainer places
    the 3-class rating label at the last token of ``step``.
    """
    problem = _problem(record)
    chosen = chosen_trajectory(record)
    steps = _steps(record)
    examples: list[dict] = []
    prefix: list[str] = []
    for i, st in enumerate(steps):
        comps = st.get("completions") or []
        # Every candidate at this position that carries a human rating is a labelled
        # (prefix, step) example -- this is where the process signal lives.
        for c in comps:
            c = c or {}
            r = c.get("rating")
            text = (c.get("text") or "").strip()
            if r is None or not text:
                continue
            examples.append({"problem": problem, "prefix": list(prefix), "step": text, "rating": int(r)})
        # Advance the prefix along the chosen trajectory and stop after the first
        # negative chosen step (supervise only up to the first mistake, per the paper).
        if i < len(chosen):
            prefix.append(chosen[i]["text"])
            if chosen[i].get("rating") == -1:
                break
        else:
            break
    return examples


def _is_train(source_name: str) -> bool:
    return "train" in source_name.lower()


# --------------------------------------------------------------------------------------
# Corpus assembly. Each ``build_*`` takes an iterator of (source_name, record) pairs and
# a grader callable ``grade(prediction, label) -> float`` (1.0 correct, 0.0 incorrect),
# so the pure logic can be tested with a fake grader and no torch import.
# --------------------------------------------------------------------------------------
def build_corpora(records: Iterable[tuple[str, dict]], grade: Callable[[str, str], float]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"sft": [], "orm": [], "prm": [], "prompts_math": [], "eval_math": []}
    seen_problems: set[str] = set()
    for source_name, record in records:
        problem = _problem(record)
        answer = _ground_truth(record)
        if not problem or not answer:
            continue
        train = _is_train(source_name)
        chosen = chosen_trajectory(record)

        # SFT + PRM + a positive ORM example come from the chosen trajectory (train only).
        if train and chosen:
            sol = solution_text(chosen, answer)
            out["sft"].append(sft_record(problem, sol))
            out["orm"].append({**sft_record(problem, sol), "outcome": int(grade(sol, answer) >= 0.5)})
            out["prm"].extend(prm_examples(record))

        # Phase-2 pre-generated whole solutions are a second ORM class (often incorrect).
        pre = (record.get("question") or {}).get("pre_generated_steps")
        if train and pre:
            pre_sol = solution_text([{"text": s} for s in pre], answer)
            out["orm"].append({**sft_record(problem, pre_sol), "outcome": int(grade(pre_sol, answer) >= 0.5)})

        # One GRPO prompt / eval item per distinct problem. Train problems feed GRPO;
        # held-out (*_test) problems are the MATH eval set.
        if problem not in seen_problems:
            seen_problems.add(problem)
            item = {"problem": problem, "answer": answer, "dataset": "math"}
            out["prompts_math" if train else "eval_math"].append(item)
    return out


# --------------------------------------------------------------------------------------
# S3 I/O (platform path only; imported lazily so --selftest stays offline/torch-free).
# --------------------------------------------------------------------------------------
def _split_uri(uri: str) -> tuple[str, str]:
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def read_prm800k(s3: Any) -> list[tuple[str, dict]]:
    from edullm_data.read import dataset_paths  # noqa: PLC0415 -- lazy so --selftest stays offline

    resolved = dataset_paths("vendor/openai-prm800k", "v1", s3=s3)
    pairs: list[tuple[str, dict]] = []
    for uri in resolved.paths:
        bucket, key = _split_uri(uri)
        name = key.rsplit("/", 1)[-1]
        raw = s3.get(bucket, key)
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                pairs.append((name, json.loads(line)))
    return pairs


def write_jsonl_s3(s3: Any, out_prefix: str, name: str, rows: list[dict]) -> str:
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    uri = out_prefix.rstrip("/") + "/" + name
    bucket, key = _split_uri(uri)
    s3.put(bucket, key, body, content_type="application/x-ndjson")
    return uri


def fetch_gsm8k() -> tuple[list[dict], list[dict]]:
    """GSM8K train/test via the Hub (Batch jobs have egress). Answer = final number."""
    from datasets import load_dataset  # noqa: PLC0415 -- lazy so --selftest stays offline

    def rows(split: str) -> list[dict]:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        out = []
        for ex in ds:
            ans = ex["answer"].split("####")[-1].strip().replace(",", "")
            out.append({"problem": ex["question"].strip(), "answer": ans, "dataset": "gsm8k"})
        return out

    return rows("train"), rows("test")


# --------------------------------------------------------------------------------------
# Self-test: pure transforms on a synthetic record, no S3 / network / torch.
# --------------------------------------------------------------------------------------
def _selftest() -> None:
    record = {
        "question": {"problem": "What is 2+3?", "ground_truth_answer": "5"},
        "label": {
            "steps": [
                {
                    "completions": [
                        {"text": "Add 2 and 3.", "rating": 1},
                        {"text": "Multiply 2 and 3.", "rating": -1},
                    ],
                    "chosen_completion": 0,
                    "human_completion": None,
                },
                {
                    "completions": [
                        {"text": "The sum is $\\boxed{5}$.", "rating": 1},
                        {"text": "The sum is 6.", "rating": 0},
                    ],
                    "chosen_completion": 0,
                    "human_completion": None,
                },
            ]
        },
    }

    chosen = chosen_trajectory(record)
    assert [c["text"] for c in chosen] == ["Add 2 and 3.", "The sum is $\\boxed{5}$."], chosen

    sol = solution_text(chosen, "5")
    assert "\\boxed{5}" in sol and sol.count(STEP_SEP) == 1, repr(sol)

    # A record whose chosen text has no boxed answer must get the boxed line appended.
    no_box = solution_text([{"text": "Some reasoning."}], "7")
    assert no_box.endswith("$\\boxed{7}$."), repr(no_box)

    prm = prm_examples(record)
    # Two positions x two rated candidates each = 4 labelled (prefix, step) examples.
    assert len(prm) == 4, prm
    assert prm[0] == {"problem": "What is 2+3?", "prefix": [], "step": "Add 2 and 3.", "rating": 1}
    assert prm[2]["prefix"] == ["Add 2 and 3."], prm[2]
    assert {e["rating"] for e in prm} == {-1, 0, 1}, prm

    # Truncation at first negative chosen step: make the first chosen step negative and
    # confirm no second-position examples are emitted.
    neg = json.loads(json.dumps(record))
    neg["label"]["steps"][0]["chosen_completion"] = 1  # the -1 candidate
    prm_neg = prm_examples(neg)
    assert all(e["prefix"] == [] for e in prm_neg), prm_neg

    # Grader is injected; use a fake that trusts a boxed "5".
    def fake_grade(prediction: str, label: str) -> float:
        return 1.0 if f"\\boxed{{{label}}}" in prediction else 0.0

    corpora = build_corpora([("phase1_train.jsonl", record)], fake_grade)
    assert len(corpora["sft"]) == 1
    assert corpora["sft"][0]["messages"][0]["role"] == "user"
    assert corpora["sft"][0]["messages"][1]["role"] == "assistant"
    assert corpora["orm"][0]["outcome"] == 1, corpora["orm"][0]
    assert len(corpora["prompts_math"]) == 1 and corpora["prompts_math"][0]["answer"] == "5"
    assert corpora["eval_math"] == []

    # A held-out (test) source routes the problem to eval, not prompts, and emits no SFT.
    corpora_test = build_corpora([("phase1_test.jsonl", record)], fake_grade)
    assert corpora_test["sft"] == [] and len(corpora_test["eval_math"]) == 1

    print("DATAPREP SELFTEST OK: transforms, truncation, routing and grader wiring verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="s3:// prefix to write the corpora to (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--selftest", action="store_true", help="run offline transform checks and exit")
    ap.add_argument("--no-gsm8k", action="store_true", help="skip the GSM8K Hub fetch")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.out or not args.out.startswith("s3://"):
        print(f"--out must be an s3:// prefix (got {args.out!r})", file=sys.stderr)
        raise SystemExit(2)

    from edullm_data.s3 import Boto3S3  # noqa: PLC0415 -- lazy so --selftest stays offline

    from open_instruct.ground_truth_utils import (  # noqa: PLC0415 -- lazy so --selftest stays offline
        GSM8KVerifier,
        MathVerifier,
    )

    math_verifier = MathVerifier()
    gsm8k_verifier = GSM8KVerifier()

    def grade_math(prediction: str, label: str) -> float:
        return float(math_verifier(tokenized_prediction=[], prediction=prediction, label=label).score)

    s3 = Boto3S3.default()

    print("reading PRM800K (vendor/openai-prm800k v1) ...", flush=True)
    records = read_prm800k(s3)
    print(f"read {len(records)} PRM800K records", flush=True)

    corpora = build_corpora(records, grade_math)

    # Prompt/eval sets from GSM8K (primary metric). Answer already extracted; store as-is.
    if not args.no_gsm8k:
        try:
            gsm_train, gsm_test = fetch_gsm8k()
            corpora["prompts_gsm8k"] = gsm_train
            corpora["eval_gsm8k"] = gsm_test
            # Sanity: GSM8KVerifier agrees with the stored answer on a few rows.
            ok = sum(
                gsm8k_verifier(tokenized_prediction=[], prediction=r["answer"], label=r["answer"]).score
                for r in gsm_test[:20]
            )
            print(f"gsm8k: {len(gsm_train)} train / {len(gsm_test)} test (verifier self-check {ok}/20)", flush=True)
        except Exception as exc:  # noqa: BLE001 -- egress is best-effort; keep the PRM800K corpora.
            print(f"WARNING: GSM8K fetch failed ({exc!r}); skipping GSM8K sets", flush=True)
            corpora.setdefault("prompts_gsm8k", [])
            corpora.setdefault("eval_gsm8k", [])

    manifest: dict[str, Any] = {"out_prefix": args.out.rstrip("/"), "counts": {}, "uris": {}}
    for name, rows in corpora.items():
        uri = write_jsonl_s3(s3, args.out, f"{name}.jsonl", rows)
        manifest["counts"][name] = len(rows)
        manifest["uris"][name] = uri
        print(f"wrote {len(rows):>7} rows -> {uri}", flush=True)

    write_jsonl_s3(s3, args.out, "MANIFEST.json", [manifest])
    print("DATAPREP OK: " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
