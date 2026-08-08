"""Check every --flag in .edullm/run.yaml against the fields the trainers actually parse.

Reads the dataclasses with ast rather than importing them, so this runs on a laptop with
no torch. Catches the one failure this smoke test is most likely to hit: a flag that looks
right, is spelled wrong, and is only discovered after a machine has been paid for.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shlex
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _module(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text())


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _own_fields(cls: ast.ClassDef) -> set[str]:
    return {
        stmt.target.id for stmt in cls.body if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def fields_of(entry: str, class_name: str, extra_modules: list[str]) -> set[str]:
    """Every field on a dataclass and on the bases it inherits, across the given modules."""
    pool: dict[str, ast.ClassDef] = {}
    for path in [entry, *extra_modules]:
        pool.update(_classes(_module(path)))

    seen: set[str] = set()
    out: set[str] = set()

    def walk(name: str) -> None:
        if name in seen or name not in pool:
            return
        seen.add(name)
        cls = pool[name]
        out.update(_own_fields(cls))
        for base in cls.bases:
            if isinstance(base, ast.Name):
                walk(base.id)

    walk(class_name)
    return out


def argparse_flags(path: str) -> set[str]:
    tree = _module(path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                    out.add(str(arg.value)[2:])
    return out


SHARED = ["open_instruct/utils.py", "open_instruct/dataset_transformation.py", "open_instruct/model_utils.py"]

# dpo_utils imports its bases from olmo_core_utils, so that module goes last and wins the
# names it shares with model_utils and dataset_transformation.
DPO_MODULES = [*SHARED, "open_instruct/dpo_utils.py", "open_instruct/olmo_core_utils.py"]

KNOWN = {
    "open_instruct/finetune.py": fields_of("open_instruct/finetune.py", "FlatArguments", SHARED)
    | fields_of("open_instruct/utils.py", "TokenizerConfig", SHARED),
    "open_instruct/dpo_tune_cache.py": fields_of("open_instruct/dpo_utils.py", "DPOExperimentConfig", DPO_MODULES)
    | fields_of("open_instruct/utils.py", "TokenizerConfig", SHARED),
    "open_instruct/merge_lora.py": argparse_flags("open_instruct/merge_lora.py"),
}

# accelerate launch's own flags, consumed before the program sees them.
LAUNCHER = {"mixed_precision", "num_processes", "config_file", "use_deepspeed", "deepspeed_multinode_launcher"}


def semantic_errors(spec: dict) -> list[str]:
    """Check cross-flag invariants that field-name validation cannot prove."""
    script = shlex.split(spec["command"])[-1]
    sft, after_sft = script.split('&& echo "=== STAGE 2/3', maxsplit=1)
    merge, dpo = after_sft.split('&& echo "=== STAGE 3/3', maxsplit=1)

    required_sft = {
        "--push_to_hub false": "SFT must not make an unauthenticated Hugging Face write",
        "--try_launch_beaker_eval_jobs false": "SFT must not launch AI2 Beaker jobs",
        "--try_auto_save_to_beaker false": "SFT must not write to an AI2 Beaker dataset",
        "--dataset_mixer_list allenai/tulu-3-sft-personas-algebra 64": (
            "the SFT mixer count, not the dead --max_train_samples field, must bound the smoke"
        ),
        't.bos_token=t.eos_token; t.save_pretrained(\\"$OUT/tokenizer\\")': (
            "the base OLMoE tokenizer must materialize its intended BOS alias before SFT"
        ),
        '--tokenizer_name_or_path "$OUT/tokenizer"': "SFT must load the tokenizer with the materialized BOS alias",
        "--mixed_precision bf16": "the platform precision guard must see bf16 in the command",
    }
    errors = [detail for text, detail in required_sft.items() if text not in sft]
    if "--max_train_samples" in sft:
        errors.append("SFT's --max_train_samples field is dead code; set the mixer count instead")
    if re.search(r"--lora_dropout 0(?:\s|$)", sft) is None:
        errors.append("SFT must use zero LoRA dropout for PEFT's OLMoE expert-parameter wrapper")
    if '--tokenizer_name_or_path "$OUT/sft"' not in merge:
        errors.append("the merge must preserve the chat template saved beside the SFT adapter")
    if re.search(r"--lora_dropout 0(?:\s|$)", dpo) is None:
        errors.append("DPO must use zero LoRA dropout for PEFT's OLMoE expert-parameter wrapper")
    if spec.get("suggested_compute") != "gpu-8xa100":
        errors.append("OLMoE-1B-7B-0125 plus its merged output does not fit the L40S host's 30GiB root disk")
    return errors


def main() -> int:
    spec = yaml.safe_load((ROOT / ".edullm/run.yaml").read_text())
    words = shlex.split(spec["command"])
    # The command is `bash -lc '<script>'`; the script is the last word.
    words = shlex.split(words[-1].replace("&&", " ").replace(";", " "))

    program = None
    bad: list[tuple[str, str]] = []
    checked = 0
    for word in words:
        if re.fullmatch(r"open_instruct/\w+\.py", word):
            program = word
            continue
        if not word.startswith("--"):
            continue
        flag = word[2:]
        if flag in LAUNCHER:
            continue
        if program is None:
            continue
        checked += 1
        if flag not in KNOWN[program]:
            bad.append((program, flag))

    print(f"checked {checked} flags across {len(KNOWN)} programs")
    for program, flag in bad:
        near = sorted(f for f in KNOWN[program] if flag.split("_")[0] in f)[:4]
        print(f"  UNKNOWN  {program}  --{flag}" + (f"   near: {near}" if near else ""))
    semantic = semantic_errors(spec)
    for detail in semantic:
        print(f"  UNSAFE   {detail}")
    if not bad and not semantic:
        print("every flag resolves to a parsed field")
        print("every smoke-test safety invariant holds")
    return 1 if bad or semantic else 0


if __name__ == "__main__":
    sys.exit(main())
