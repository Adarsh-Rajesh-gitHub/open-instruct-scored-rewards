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
        stmt.target.id
        for stmt in cls.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
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
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
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
    "open_instruct/dpo_tune_cache.py": fields_of(
        "open_instruct/dpo_utils.py", "DPOExperimentConfig", DPO_MODULES
    )
    | fields_of("open_instruct/utils.py", "TokenizerConfig", SHARED),
    "open_instruct/merge_lora.py": argparse_flags("open_instruct/merge_lora.py"),
}

# accelerate launch's own flags, consumed before the program sees them.
LAUNCHER = {"mixed_precision", "num_processes", "config_file", "use_deepspeed", "deepspeed_multinode_launcher"}


def main() -> int:
    spec = yaml.safe_load((ROOT / ".edullm/run.yaml").read_text())
    words = shlex.split(spec["command"])
    # The command is `bash -lc '<script>'`; the script is the last word.
    words = shlex.split(words[-1].replace("&&", " ").replace(";", " "))

    program = None
    bad: list[tuple[str, str]] = []
    checked = 0
    for i, word in enumerate(words):
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
    if not bad:
        print("every flag resolves to a parsed field")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
