"""Stages 3-4 S3 orchestration: train one reward model (ORM or PRM) around ``reward_modeling_scored.py``.

Like ``run_sft.py``, this does the S3 plumbing the trainer itself does not (the trainer reads and
writes only local paths):

  1. download the Stage-2 SFT model directory (``--sft-uri``) to local scratch;
  2. download the Stage-1 data file (``--data-uri``: ``orm.jsonl`` or ``prm.jsonl``);
  3. ``accelerate launch`` ``reward_modeling_scored.py --rm-type {orm,prm}`` across every visible GPU;
  4. upload the trained RM directory to ``$EDULLM_CHECKPOINT_DIR`` for Stage 5 (best-of-N) and
     Stage 6 (the GRPO reward bridge) to consume.

Run (per RM) as an eduLLM ``-train`` GPU job (W&B auto-wired by the GPU role)::

    python projects/prm_vs_orm/run_rm.py --rm-type orm \\
        --sft-uri  s3://.../runs/<stage2_run>/checkpoints/ \\
        --data-uri s3://.../runs/<stage1_run>/checkpoints/orm.jsonl \\
        --out      "$EDULLM_CHECKPOINT_DIR" --dtype bfloat16

``--selftest`` exercises the pure URI + command-builder helpers offline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RM_TRAINER = "projects/prm_vs_orm/reward_modeling_scored.py"


def _split_uri(uri: str) -> tuple[str, str]:
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


def build_accelerate_cmd(
    *,
    num_gpus: int,
    rm_type: str,
    base_dir: str,
    data_file: str,
    output_dir: str,
    max_seq_length: int,
    epochs: float,
    lr: float,
    per_device_batch: int,
    grad_accum: int,
    max_examples: int,
    with_tracking: bool,
) -> list[str]:
    """Assemble the ``accelerate launch`` argv for the RM trainer (bf16 data-parallel)."""
    cmd: list[str] = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_processes", str(max(1, num_gpus)),
        RM_TRAINER,
        "--rm-type", rm_type,
        "--model-name-or-path", base_dir,
        "--data-file", data_file,
        "--output-dir", output_dir,
        "--max-seq-length", str(max_seq_length),
        "--num-train-epochs", str(epochs),
        "--learning-rate", str(lr),
        "--per-device-batch-size", str(per_device_batch),
        "--gradient-accumulation-steps", str(grad_accum),
        "--dtype", "bfloat16",
    ]
    if max_examples > 0:
        cmd += ["--max-examples", str(max_examples)]
    if with_tracking:
        cmd += ["--with-tracking", "--wandb-entity", "eduLLM", "--wandb-project-name", f"prm-vs-orm-{rm_type}"]
    return cmd


_ERR_MARKER = "RMERR| "  # kept in sync with reward_modeling_scored.py's marker


def _run_and_surface_errors(cmd: list[str]) -> None:
    """Run the accelerate launch, streaming output, and on failure re-print the child's real
    traceback LAST — the RM-stage analog of run_sft.py's helper.

    torch.distributed.elastic prints its per-rank exit-code summary *after* the failing rank's
    traceback, so the ~50-line tail ``edullm logs`` returns shows only teardown boilerplate. We
    tee the child's combined output and, on non-zero exit, pull out the ``RMERR|``-marked lines
    (the real rank-0 traceback the trainer emitted) and print them last so they land in that
    tail. Falls back to the raw output tail if no marker is present.
    """
    log_path = Path("/tmp/rm_launch.log")
    with log_path.open("wb") as f:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            f.write(chunk)
        rc = proc.wait()
    if rc != 0:
        lines = log_path.read_text(errors="replace").splitlines()
        marked = [ln for ln in lines if _ERR_MARKER in ln]
        print("\n=================== RM REAL ERROR (rank-0 traceback) ===================", flush=True)
        if marked:
            print("\n".join(marked[-60:]), flush=True)
        else:
            print("(no RMERR marker found — last 40 lines of child output:)", flush=True)
            print("\n".join(lines[-40:]), flush=True)
        print("========================================================================", flush=True)
        raise SystemExit(rc)


def _detect_gpus() -> int:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.device_count()
    except (ImportError, RuntimeError):
        return 0


def _selftest() -> None:
    assert _split_uri("s3://b/a/c/") == ("b", "a/c/")
    assert _rel_key("t/checkpoints/model.safetensors", "t/checkpoints/") == "model.safetensors"
    cmd = build_accelerate_cmd(
        num_gpus=1, rm_type="prm", base_dir="/sft", data_file="/d/prm.jsonl", output_dir="/o",
        max_seq_length=1024, epochs=1, lr=1e-5, per_device_batch=8, grad_accum=4, max_examples=0,
        with_tracking=True,
    )
    assert cmd[:2] == ["accelerate", "launch"] and RM_TRAINER in cmd
    assert cmd[cmd.index("--rm-type") + 1] == "prm"
    assert cmd[cmd.index("--model-name-or-path") + 1] == "/sft"
    assert "--with-tracking" in cmd and cmd[cmd.index("--wandb-entity") + 1] == "eduLLM"
    cmd2 = build_accelerate_cmd(
        num_gpus=4, rm_type="orm", base_dir="/sft", data_file="/d/orm.jsonl", output_dir="/o",
        max_seq_length=512, epochs=1, lr=1e-5, per_device_batch=16, grad_accum=1, max_examples=100,
        with_tracking=False,
    )
    assert cmd2[cmd2.index("--num_processes") + 1] == "4"
    assert cmd2[cmd2.index("--max-examples") + 1] == "100" and "--with-tracking" not in cmd2
    print("RUN_RM SELFTEST OK: uri helpers + accelerate command builder verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rm-type", choices=["orm", "prm"])
    ap.add_argument("--sft-uri", help="s3:// prefix of the Stage-2 SFT model")
    ap.add_argument("--data-uri", help="s3:// URI of orm.jsonl / prm.jsonl")
    ap.add_argument("--out", help="s3:// prefix to write the RM to (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--local-sft", default="/tmp/sft_model")
    ap.add_argument("--local-data", default=None)
    ap.add_argument("--local-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    for name, val in (("--rm-type", args.rm_type), ("--sft-uri", args.sft_uri),
                      ("--data-uri", args.data_uri), ("--out", args.out)):
        if not val:
            print(f"{name} is required", file=sys.stderr)
            raise SystemExit(2)
    for name, val in (("--sft-uri", args.sft_uri), ("--data-uri", args.data_uri), ("--out", args.out)):
        if not val.startswith("s3://"):
            print(f"{name} must be an s3:// URI (got {val!r})", file=sys.stderr)
            raise SystemExit(2)

    local_data = args.local_data or f"/tmp/rm_data/{args.rm_type}.jsonl"
    local_out = args.local_out or f"/tmp/rm_out_{args.rm_type}"

    from edullm_data.s3 import Boto3S3  # noqa: PLC0415

    s3 = Boto3S3.default()

    print(f"downloading SFT model {args.sft_uri} -> {args.local_sft} ...", flush=True)
    nbase = download_prefix(s3, args.sft_uri, args.local_sft)
    print(f"downloaded {nbase} SFT files", flush=True)

    print(f"downloading data {args.data_uri} -> {local_data} ...", flush=True)
    download_file(s3, args.data_uri, local_data)

    num_gpus = _detect_gpus()
    print(f"visible GPUs: {num_gpus}", flush=True)

    cmd = build_accelerate_cmd(
        num_gpus=num_gpus,
        rm_type=args.rm_type,
        base_dir=args.local_sft,
        data_file=local_data,
        output_dir=local_out,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        lr=args.lr,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        max_examples=args.max_examples,
        with_tracking=not args.no_tracking,
    )
    print("launching:", " ".join(cmd), flush=True)
    _run_and_surface_errors(cmd)

    print(f"uploading RM {local_out} -> {args.out} ...", flush=True)
    uploaded = upload_dir(s3, local_out, args.out)
    print(f"RM RUN OK: rm_type={args.rm_type} rm={args.out.rstrip('/')} files={len(uploaded)}", flush=True)


if __name__ == "__main__":
    main()
