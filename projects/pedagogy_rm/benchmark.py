"""Measure whether training the tutor cost it any academic ability.

    python projects/pedagogy_rm/benchmark.py --adapter '' --tag base
    python projects/pedagogy_rm/benchmark.py --adapter output/.../step_200 --tag arm_e

THE QUESTION. The policy was optimised for one narrow thing - the shape of a single tutoring turn
- for 200 steps against a reward that says nothing about whether it can still do arithmetic or
recall a fact. Reinforcement learning on a narrow objective is well known to cost capability
elsewhere, and nothing measured so far would notice: every dimension in the reward is about how a
turn is phrased, and the blind evaluation only ever compared tutoring against tutoring.

So this asks the policy to answer questions rather than teach them, on benchmarks it was never
trained on, and compares against the untrained model.

WHY NOT lm-eval. It was tried first, and lm-eval 0.4.9 is incompatible with the transformers
version this environment pins (it expects AutoModelForVision2Seq). Repinning transformers in an
environment where vLLM, DeepSpeed and the training loop all work is a worse trade than eighty lines
of multiple-choice scoring, which is all that is needed here.

GENERATION RATHER THAN LOGLIKELIHOOD SCORING, deliberately. Ranking options by their loglikelihood
measures the base model underneath the adapter; asking for a letter and parsing it measures what
the policy actually does when prompted, which is the thing that could have degraded. It also means
a policy that has forgotten how to answer at all - rather than merely answering wrongly - shows up
as unparseable rather than as chance-level accuracy, and those are different failures.

The prompt is deliberately plain, with no tutoring framing. A policy trained to teach might
otherwise respond to a question with a question, which would be a real finding but a different one.
"""

from __future__ import annotations

import argparse
import json
import os
import re


def load_arc(limit: int) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    rows = []
    data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for item in data:
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        if item["answerKey"] not in labels:
            continue
        rows.append({"question": item["question"], "options": texts,
                     "gold": labels.index(item["answerKey"])})
        if len(rows) >= limit:
            break
    return rows


def load_mmlu_stem(limit: int) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    # A stem slice rather than all 57 subjects: the training questions were maths and science at
    # school level, so this is where a change is most likely to show and cheapest to detect.
    subjects = ["high_school_mathematics", "high_school_biology", "high_school_chemistry",
                "high_school_physics", "elementary_mathematics"]
    rows = []
    for subject in subjects:
        data = load_dataset("cais/mmlu", subject, split="test")
        for item in data:
            rows.append({"question": item["question"], "options": item["choices"],
                         "gold": int(item["answer"])})
            if len(rows) >= limit:
                return rows
    return rows


def load_gsm8k(limit: int) -> list[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    rows = []
    for item in load_dataset("openai/gsm8k", "main", split="test"):
        answer = item["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"question": item["question"], "options": None, "gold": answer})
        if len(rows) >= limit:
            break
    return rows


LOADERS = {"arc_challenge": load_arc, "mmlu_stem": load_mmlu_stem, "gsm8k": load_gsm8k}
LETTERS = "ABCDEFGH"


def prompt_for(row: dict) -> str:
    # LET IT SHOW ITS WORKING, rather than demanding a bare answer. The first version of this
    # asked for "only the final numeric answer, with no working" and capped generation at 24
    # tokens; an instruction-tuned model reasons anyway, so half the GSM8K generations were cut
    # off mid-sentence and the measured accuracy was 1.8% against a published ~67%. That number
    # was a property of the harness, not the model. Asking for the answer on a final line, with
    # room to reach it, measures the model instead.
    if row["options"] is None:
        return (f"{row['question']}\n\n"
                "Reason briefly, then end with a final line of the form 'Answer: <number>'.")
    options = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(row["options"]))
    return (f"{row['question']}\n\n{options}\n\n"
            "Reason briefly, then end with a final line of the form 'Answer: <letter>'.")


NUMBER = r"-?\d[\d,]*\.?\d*"


def parse(text: str, row: dict) -> str | None:
    """Pull the model's answer out, preferring where it says so over where it merely mentions one.

    Ordered rather than a single pattern, because the two failure modes pull in opposite
    directions: taking the last number anywhere catches a model that reasons past its own
    conclusion, but misreads one that ends on a restated quantity. Looking for the declared
    answer first and only then falling back keeps both.
    """
    text = (text or "").strip()
    if not text:
        return None
    if row["options"] is None:
        body = text.replace("$", "").replace("\\", "")
        for pattern in (rf"(?:answer|total|result)\s*(?:is|:|=)\s*\**\s*({NUMBER})",
                        rf"boxed\{{\s*({NUMBER})"):
            found = re.findall(pattern, body, re.IGNORECASE)
            if found:
                return found[-1].replace(",", "").rstrip(".")
        found = re.findall(NUMBER, body)
        return found[-1].replace(",", "").rstrip(".") if found else None

    valid = LETTERS[: len(row["options"])]
    for pattern in (rf"answer\s*(?:is|:|=)\s*\**\s*\(?([{valid}])\b",
                    rf"^\**\(?([{valid}])[.):]", rf"\*\*([{valid}])\*\*"):
        found = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if found:
            return found[-1].upper()
    # Last resort: the model named an option's text without ever labelling it with its letter.
    # Only the tail is searched, and only an unambiguous hit counts: options are often short
    # ("4", "12"), so a substring search over the whole response would match the restated
    # question as readily as the conclusion. Requiring exactly one option in the closing words
    # is what makes this safe for short options rather than only for long ones.
    tail = text[-120:].lower()
    named = [i for i, o in enumerate(row["options"])
             # Boundaries that let a sentence-final "12." match while "12.5" and "120" do not:
             # a trailing full stop is punctuation, a trailing digit is a different number.
             if o and re.search(rf"(?<![\w.]){re.escape(o.lower())}(?!\w)(?!\.\d)", tail)]
    if len(named) == 1:
        return LETTERS[named[0]]
    hit = re.search(rf"\b([{valid}])\b", text[:60])
    return hit.group(1).upper() if hit else None


def score_by_loglikelihood(llm, tokenizer, rows: list[dict], lora, args, task: str) -> dict:
    """Pick the option the model finds most likely, instead of asking it to name one.

    WHY THIS EXISTS ALONGSIDE THE GENERATION PATH. Asked to reason and then state a letter, this
    model failed to reach a parseable answer on 30% of MMLU stem items - 86 of 400 were still
    reasoning when the token budget ran out. That is fatal to the comparison rather than merely
    noisy: a policy trained to be more verbose loses more items to truncation, so a drop in
    accuracy cannot be told apart from a change in verbosity. Base scored 41.0% and the highest
    learning-rate arm 36.2%, with 119 and 129 items unparseable respectively, and no amount of
    staring at those numbers says which effect produced the gap.

    Scoring the letter directly removes the failure mode instead of budgeting around it: every
    item yields a comparison, the count is fixed at four forward passes per question, and nothing
    depends on the model's willingness to follow a format. This is also what standard harnesses
    do for MMLU, so the absolute numbers are comparable to published ones.

    Token ids are passed through rather than strings, because re-tokenising context+letter can
    merge the boundary and shift which positions belong to the answer.
    """
    from vllm import SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415

    prompts, spans = [], []
    for row in rows:
        options = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(row["options"]))
        context = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{row['question']}\n\n{options}\n\n"
                                         "Answer with the letter of the correct option."}],
            tokenize=False, add_generation_prompt=True)
        ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
        for i in range(len(row["options"])):
            full = tokenizer(context + LETTERS[i], add_special_tokens=False).input_ids
            prompts.append(TokensPrompt(prompt_token_ids=full))
            spans.append(len(ctx_ids))

    # max_tokens=1 because nothing is being generated; the scores come from prompt_logprobs.
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
                           lora_request=lora)

    right, k, per_item = 0, 0, []
    for row in rows:
        totals = []
        for _ in row["options"]:
            out, start = outputs[k], spans[k]
            k += 1
            total = 0.0
            for pos, entry in enumerate(out.prompt_logprobs or []):
                if pos >= start and entry:
                    ident = out.prompt_token_ids[pos]
                    if ident in entry:
                        total += entry[ident].logprob
            totals.append(total)
        hit = max(range(len(totals)), key=totals.__getitem__) == row["gold"]
        right += hit
        per_item.append(int(hit))

    print(f"  {task:<15} {right}/{len(rows)} = {right/len(rows):.1%}   (loglikelihood, 0 unparseable)")
    return {"n": len(rows), "correct": right, "accuracy": right / len(rows), "unparseable": 0,
            "truncated": 0, "mode": "loglikelihood", "per_item": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--adapter", default="", help="a PEFT adapter directory; empty means base")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tasks", default="arc_challenge,mmlu_stem,gsm8k")
    parser.add_argument("--limit", type=int, default=400, help="items per task")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mc-mode", choices=["loglikelihood", "generate"], default="loglikelihood",
                        help="how to answer multiple-choice items; gsm8k always generates")
    parser.add_argument("--vllm-util", type=float, default=0.70)
    parser.add_argument("--out", default="data/benchmarks")
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    tasks = [t for t in args.tasks.split(",") if t]
    loaded = {t: LOADERS[t](args.limit) for t in tasks}
    for t, rows in loaded.items():
        print(f"{t}: {len(rows)} items")

    engine_kwargs = {}
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest  # noqa: PLC0415

        engine_kwargs = {"enable_lora": True, "max_lora_rank": args.lora_rank}
        lora = LoRARequest("policy", 1, args.adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.policy)
    llm = LLM(model=args.policy, gpu_memory_utilization=args.vllm_util,
              max_model_len=2048, **engine_kwargs)
    # Greedy: this is a capability measurement, and sampling would add variance that has nothing
    # to do with the thing being compared.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    results = {}
    for task, rows in loaded.items():
        if rows and rows[0]["options"] is not None and args.mc_mode == "loglikelihood":
            results[task] = score_by_loglikelihood(llm, tokenizer, rows, lora, args, task)
            continue
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(r)}], tokenize=False, add_generation_prompt=True)
            for r in rows]
        outputs = llm.generate(prompts, sampling, lora_request=lora)
        right = unparseable = truncated = 0
        raw, per_item = [], []
        for row, out in zip(rows, outputs, strict=True):
            text = out.outputs[0].text
            got = parse(text, row)
            want = row["gold"] if row["options"] is None else LETTERS[row["gold"]]
            if got is None:
                unparseable += 1
            else:
                right += got == want
            per_item.append(int(got is not None and got == want))
            # A generation stopped by the token budget rather than by the model is the signature
            # of the harness bug this replaced, so it is counted rather than left to be inferred.
            truncated += out.outputs[0].finish_reason == "length"
            raw.append({"got": got, "want": want, "finish": out.outputs[0].finish_reason,
                        "text": text[:600]})
        results[task] = {"n": len(rows), "correct": right, "accuracy": right / len(rows),
                         "unparseable": unparseable, "truncated": truncated,
                         "mode": "generate", "per_item": per_item}
        print(f"  {task:<15} {right}/{len(rows)} = {right/len(rows):.1%}"
              f"   ({unparseable} unparseable, {truncated} hit the token limit)")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.tag}.{task}.raw.jsonl"), "w") as handle:
            for item in raw:
                handle.write(json.dumps(item) + "\n")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w") as handle:
        json.dump({"tag": args.tag, "policy": args.policy, "adapter": args.adapter,
                   "limit": args.limit, "results": results}, handle, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
