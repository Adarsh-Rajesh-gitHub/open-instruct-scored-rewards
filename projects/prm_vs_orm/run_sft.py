"""Stage 2 for the PRM-vs-ORM experiment: supervised fine-tune the HF base on ``sft.jsonl``.

open-instruct's ``finetune.py`` neither reads nor writes S3 and expects local paths, so this
wrapper does the S3 plumbing around it (the same job-role pattern as ``convert_base.py`` and
``data_prep.py``):

  1. download the Stage-0 HF base model directory (``--hf-base-uri``) to local scratch;
  2. download the Stage-1 ``sft.jsonl`` (``--sft-uri``) to local scratch;
  3. ``accelerate launch`` ``projects/prm_vs_orm/_sft_launch.py`` (the finetune shim) across
     every visible GPU, training on the local jsonl via ``--dataset_mixer_list <file> 1.0``;
  4. upload the trained model directory to ``$EDULLM_CHECKPOINT_DIR`` for Stages 3-5 (RM
     training, best-of-N) to consume.

**Format consistency is a correctness requirement.** SFT, RM training, best-of-N and GRPO must
all use ONE prompt template + tokenizer. We use the ``tulu`` chat template (plain
``<|user|>``/``<|assistant|>`` turns; no injected persona or ``<|im_start|>`` specials that the
dolma2 vocab lacks) over the ``{"messages":[...]}`` records ``data_prep.py`` emitted, and the
dolma2 tokenizer bundled with the base. ``--add-bos auto`` prepends BOS only if the tokenizer
actually defines one.

Run as an eduLLM ``-train`` GPU job (W&B is auto-wired by the GPU execution role)::

    python projects/prm_vs_orm/run_sft.py \
        --hf-base-uri s3://.../runs/<stage0_run>/checkpoints/ \
        --sft-uri     s3://.../runs/<stage1_run>/checkpoints/sft.jsonl \
        --out         "$EDULLM_CHECKPOINT_DIR" --dtype bfloat16

``--selftest`` exercises the pure URI + command-builder helpers offline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SFT_LAUNCHER = "projects/prm_vs_orm/_sft_launch.py"


def _split_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/key/prefix`` -> ``(bucket, "key/prefix")``."""
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def _rel_key(key: str, prefix: str) -> str:
    prefix = prefix.rstrip("/") + "/"
    return key[len(prefix) :] if key.startswith(prefix) else key


def download_prefix(s3: Any, uri: str, local_dir: str) -> int:
    bucket, prefix = _split_uri(uri)
    n = 0
    for obj in s3.list(bucket, prefix):
        key = obj["key"]
        rel = _rel_key(key, prefix)
        if not rel or key.endswith("/"):
            continue
        dst = Path(local_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(s3.get(bucket, key))
        n += 1
    if n == 0:
        raise RuntimeError(f"no objects found under {uri}")
    return n


def download_file(s3: Any, uri: str, local_path: str) -> None:
    bucket, key = _split_uri(uri)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).write_bytes(s3.get(bucket, key))


def upload_dir(s3: Any, local_dir: str, out_uri: str) -> list[str]:
    bucket, prefix = _split_uri(out_uri)
    prefix = prefix.rstrip("/")
    uploaded: list[str] = []
    for p in sorted(Path(local_dir).rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}"
            s3.put_file(bucket, key, str(p))
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def tokenizer_has_bos(model_dir: str) -> bool:
    """True if the tokenizer bundled in ``model_dir`` defines a BOS token.

    Reads ``tokenizer_config.json`` / ``special_tokens_map.json`` directly (no transformers
    import) so the decision is available before ``--add-bos`` is baked into the launch command.
    """
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        p = Path(model_dir) / name
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        bos = cfg.get("bos_token")
        if bos:  # a non-empty string, or a dict with content
            return True
    return False


def build_accelerate_cmd(
    *,
    num_gpus: int,
    base_dir: str,
    sft_file: str,
    output_dir: str,
    add_bos: bool,
    chat_template: str,
    max_seq_length: int,
    epochs: float,
    lr: float,
    per_device_batch: int,
    grad_accum: int,
    warmup_ratio: float,
    wandb_project: str,
    with_tracking: bool,
) -> list[str]:
    """Assemble the ``accelerate launch`` argv for the finetune shim (bf16 data-parallel)."""
    cmd: list[str] = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_processes", str(max(1, num_gpus)),
        SFT_LAUNCHER,
        "--model_name_or_path", base_dir,
        "--tokenizer_name_or_path", base_dir,
        "--dataset_mixer_list", sft_file, "1.0",
        "--max_seq_length", str(max_seq_length),
        "--per_device_train_batch_size", str(per_device_batch),
        "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", str(lr),
        "--lr_scheduler_type", "linear",
        "--warmup_ratio", str(warmup_ratio),
        "--num_train_epochs", str(epochs),
        "--chat_template_name", chat_template,
        "--reduce_loss", "mean",
        "--output_dir", output_dir,
        "--do_not_randomize_output_dir",
        "--dataset_skip_cache",
        "--seed", "123",
        "--logging_steps", "1",
    ]
    if add_bos:
        cmd.append("--add_bos")
    if with_tracking:
        cmd += [
            "--with_tracking",
            "--report_to", "wandb",
            "--wandb_entity", "eduLLM",
            "--wandb_project_name", wandb_project,
        ]
    return cmd


def _detect_gpus() -> int:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        # absent torch or no usable driver => treat as CPU / single process
        return 0


def _selftest() -> None:
    assert _split_uri("s3://b/a/c/") == ("b", "a/c/")
    assert _rel_key("teams/t/checkpoints/config.json", "teams/t/checkpoints/") == "config.json"
    cmd = build_accelerate_cmd(
        num_gpus=4, base_dir="/tmp/base_hf", sft_file="/tmp/d/sft.jsonl", output_dir="/tmp/out",
        add_bos=True, chat_template="tulu", max_seq_length=2048, epochs=3, lr=1e-5,
        per_device_batch=16, grad_accum=1, warmup_ratio=0.03, wandb_project="prm-vs-orm-sft",
        with_tracking=True,
    )
    assert cmd[:2] == ["accelerate", "launch"]
    assert "--num_processes" in cmd and cmd[cmd.index("--num_processes") + 1] == "4"
    assert SFT_LAUNCHER in cmd
    # dataset passed as (path, proportion) pair
    i = cmd.index("--dataset_mixer_list")
    assert cmd[i + 1] == "/tmp/d/sft.jsonl" and cmd[i + 2] == "1.0"
    assert "--add_bos" in cmd and "--do_not_randomize_output_dir" in cmd
    assert "--with_tracking" in cmd and cmd[cmd.index("--wandb_entity") + 1] == "eduLLM"
    # add_bos omitted when False
    cmd2 = build_accelerate_cmd(
        num_gpus=1, base_dir="/b", sft_file="/f.jsonl", output_dir="/o", add_bos=False,
        chat_template="tulu", max_seq_length=1024, epochs=1, lr=1e-5, per_device_batch=8,
        grad_accum=2, warmup_ratio=0.0, wandb_project="p", with_tracking=False,
    )
    assert "--add_bos" not in cmd2 and "--with_tracking" not in cmd2
    print("SFT SELFTEST OK: uri helpers + accelerate command builder verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-base-uri", help="s3:// prefix of the Stage-0 HF base model")
    ap.add_argument("--sft-uri", help="s3:// URI of the Stage-1 sft.jsonl")
    ap.add_argument("--out", help="s3:// prefix to write the SFT model to (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--chat-template", default="tulu", help="open-instruct chat template name")
    ap.add_argument("--add-bos", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-device-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--wandb-project", default="prm-vs-orm-sft")
    ap.add_argument("--no-tracking", action="store_true", help="disable W&B (e.g. CPU smoke)")
    ap.add_argument("--dtype", default="bfloat16", help="named for the precision guard")
    ap.add_argument("--local-base", default="/tmp/base_hf")
    ap.add_argument("--local-sft", default="/tmp/sft_data/sft.jsonl")
    ap.add_argument("--local-out", default="/tmp/sft_out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name, val in (("--hf-base-uri", args.hf_base_uri), ("--sft-uri", args.sft_uri), ("--out", args.out)):
        if not val or not val.startswith("s3://"):
            print(f"{name} must be an s3:// URI (got {val!r})", file=sys.stderr)
            raise SystemExit(2)

    from edullm_data.s3 import Boto3S3  # noqa: PLC0415
    s3 = Boto3S3.default()

    print(f"downloading base {args.hf_base_uri} -> {args.local_base} ...", flush=True)
    nbase = download_prefix(s3, args.hf_base_uri, args.local_base)
    print(f"downloaded {nbase} base files", flush=True)

    print(f"downloading sft data {args.sft_uri} -> {args.local_sft} ...", flush=True)
    download_file(s3, args.sft_uri, args.local_sft)

    if args.add_bos == "auto":
        add_bos = tokenizer_has_bos(args.local_base)
        print(f"add_bos auto-detected as {add_bos} (tokenizer bos_token present={add_bos})", flush=True)
    else:
        add_bos = args.add_bos == "yes"

    num_gpus = _detect_gpus()
    print(f"visible GPUs: {num_gpus}", flush=True)

    cmd = build_accelerate_cmd(
        num_gpus=num_gpus,
        base_dir=args.local_base,
        sft_file=args.local_sft,
        output_dir=args.local_out,
        add_bos=add_bos,
        chat_template=args.chat_template,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        lr=args.lr,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        wandb_project=args.wandb_project,
        with_tracking=not args.no_tracking,
    )
    print("launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    print(f"uploading SFT model {args.local_out} -> {args.out} ...", flush=True)
    uploaded = upload_dir(s3, args.local_out, args.out)
    print(f"SFT OK: sft_model={args.out.rstrip('/')} files={len(uploaded)}", flush=True)


if __name__ == "__main__":
    main()
