"""Cache the RLVR datasets, and check they carry a verifier open-instruct actually has.

    python projects/rlvr_verifiable/prefetch.py --mix math
    python projects/rlvr_verifiable/prefetch.py --mix all --report

WHY THIS EXISTS AND IS NOT OPTIONAL. The Slurm wrapper sets HF_HUB_OFFLINE=1, deliberately - a run
that silently downloads a different revision is not reproducible - so anything not already in
$HF_HOME fails at startup rather than being fetched. This is the step that puts it there, and it
runs on the login node where there is a network.

WHY IT ALSO INSPECTS RATHER THAN JUST DOWNLOADING. Every row names its own verifier in its `dataset`
field, and `apply_verifiable_reward` skips rows whose name is not registered with nothing but a
warning buried in the log. A mix that is 30% unregistered trains on 70% of its data and reports a
reward that looks merely mediocre. Checking the names against the registry up front turns that into
an error before a GPU is spent.
"""

from __future__ import annotations

import argparse
import collections

# Name -> the domain it belongs to, for the report. Anything registered but absent here still
# passes; this is for grouping, not validation.
DOMAIN = {
    "gsm8k": "maths", "math": "maths", "strict_math": "maths",
    "code": "code", "code_stdio": "code",
    "string_f1": "factual", "string_matcher": "factual", "re_search_f1": "factual",
    "ifeval": "instruction following", "ifeval_old": "instruction following",
    "max_length": "length", "up_to_max_length": "length", "passthrough": "none",
}

MIXES = {
    # The mix the tulu3 8B RLVR run used, so there is a published result to compare against. Carries
    # gsm8k, math and ifeval rows, which means one download exercises three verifiers.
    "math": ["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"],
    "math_only": ["allenai/RLVR-MATH", "allenai/RLVR-GSM"],
    "code": [
        "allenai/rlvr-code-data-python-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered"
    ],
    # Ai2's general mix. Whether it is usable for the factual domain depends on which verifiers its
    # rows name - run with --report and read the breakdown before trusting it.
    "general": ["allenai/rlvr_general_mix-keyword-filtered"],
}


def registered_names() -> set[str]:
    """The verifier names open-instruct will actually dispatch on."""
    from open_instruct.ground_truth_utils import build_all_verifiers  # noqa: PLC0415

    class Args:
        code_api_url = ""
        code_max_execution_time = 1.0
        code_pass_rate_reward_threshold = 0.0
        code_apply_perf_penalty = False
        llm_judge_model = ""
        llm_judge_max_tokens = 0
        llm_judge_max_context_length = 0
        llm_judge_timeout = 0
        llm_judge_temperature = 0.0
        seed = 1
        max_length_verifier_max_length = 0
        remap_verifier = None

    return set(build_all_verifiers(Args()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mix", default="math", choices=[*MIXES, "all"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--report", action="store_true", help="tabulate the verifiers each mix uses")
    parser.add_argument("--sample", type=int, default=2000, help="rows to inspect for the report")
    args = parser.parse_args()

    import datasets  # noqa: PLC0415

    known = registered_names()
    names = [n for m in (MIXES if args.mix == "all" else {args.mix: MIXES[args.mix]}) for n in MIXES[m]]

    problems = []
    for name in names:
        print(f"\n=== {name} ===")
        try:
            ds = datasets.load_dataset(name, split=args.split)
        except Exception as exc:  # noqa: BLE001 - one unreachable dataset should not stop the rest
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            problems.append((name, "could not load"))
            continue
        print(f"  cached, {len(ds)} rows, columns {sorted(ds.column_names)}")

        for required in ("messages", "ground_truth", "dataset"):
            if required not in ds.column_names:
                print(f"  MISSING COLUMN {required!r} - grpo_fast.py needs it")
                problems.append((name, f"missing {required}"))

        if args.report and "dataset" in ds.column_names:
            counts: collections.Counter = collections.Counter()
            for row in ds.select(range(min(args.sample, len(ds)))):
                value = row["dataset"]
                for one in value if isinstance(value, list) else [value]:
                    counts[str(one).lower()] += 1
            total = sum(counts.values())
            print(f"  verifiers named, in the first {min(args.sample, len(ds))} rows:")
            for verifier, n in counts.most_common():
                mark = "ok " if verifier in known else "NOT REGISTERED"
                print(f"    {verifier:<24} {n:>6} ({n/total:>5.1%})  {mark}"
                      f"  {DOMAIN.get(verifier, '')}")
            unknown = sum(n for v, n in counts.items() if v not in known)
            if unknown:
                print(f"  WARNING: {unknown/total:.1%} of rows name a verifier that is not "
                      f"registered; those rows would be silently skipped and score zero")
                problems.append((name, f"{unknown/total:.0%} unregistered"))

    print("\n" + "=" * 60)
    if problems:
        print("problems found:")
        for name, why in problems:
            print(f"  {name}: {why}")
        raise SystemExit(1)
    print("all datasets cached, columns present, every verifier registered")


if __name__ == "__main__":
    main()
