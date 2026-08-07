"""Stage 0 for the PRM-vs-ORM experiment: convert the base OLMo-core checkpoint to HF.

The base policy is the plain-attention **Olmo3 370M @ 10B tokens** checkpoint
(``olmo3-370m/run-10b-equal/step12716``), an OLMo-core distributed checkpoint (DCP).
It has been staged into the outputs bucket (readable by the GPU/CPU job role) at a
``s3://.../checkpoints/dcp/`` prefix, because the original ``edullm-checkpoints`` bucket
is not in any job role's grant.

This job:

  1. downloads the staged DCP directory to local scratch (``edullm_data``'s S3 client,
     the same job-role path as ``data_prep.py``);
  2. rebuilds the model from the checkpoint's own experiment ``config.json`` and loads the
     DCP weights with OLMo-core's ``convert_checkpoint_to_hf`` (CPU), which also **validates**
     that the converted HF logits match the original — the real correctness gate;
  3. uploads the resulting HF model directory (``config.json`` + ``model.safetensors`` +
     tokenizer) to ``$EDULLM_CHECKPOINT_DIR`` for Stage 2 (SFT) to consume.

Run as an eduLLM ``-train`` CPU job:

    python projects/prm_vs_orm/convert_base.py \
        --dcp-uri  s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/base-olmo3-370m-step12716/checkpoints/dcp/ \
        --out      "$EDULLM_CHECKPOINT_DIR"

Output must be local before upload: HF ``tokenizer.save_pretrained`` cannot write to an
``s3://`` path, so we convert into a local directory and upload the tree ourselves.

``--selftest`` exercises the pure URI/relative-path helpers offline (no S3, no torch).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

#: The dolma2 tokenizer the base was trained with (config's ``dataset.tokenizer.identifier``).
#: Passed explicitly so the converted HF dir always carries a tokenizer even though the staged
#: checkpoint has no sibling ``tokenizer/`` directory for the converter to pick up.
DEFAULT_TOKENIZER = "allenai/dolma2-tokenizer"


def _split_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/key/prefix`` -> ``(bucket, "key/prefix")``."""
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def _rel_key(key: str, prefix: str) -> str:
    """The portion of an object key below ``prefix`` (a POSIX relative path)."""
    prefix = prefix.rstrip("/") + "/"
    return key[len(prefix) :] if key.startswith(prefix) else key


def download_prefix(s3: Any, uri: str, local_dir: str) -> int:
    """Download every object under ``uri`` into ``local_dir``, preserving structure."""
    bucket, prefix = _split_uri(uri)
    objs = s3.list(bucket, prefix)
    n = 0
    for obj in objs:
        key = obj["key"]
        rel = _rel_key(key, prefix)
        if not rel or key.endswith("/"):
            continue  # skip the prefix placeholder itself
        dst = Path(local_dir) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(s3.get(bucket, key))
        n += 1
    if n == 0:
        raise RuntimeError(f"no objects found under {uri}")
    return n


def upload_dir(s3: Any, local_dir: str, out_uri: str) -> list[str]:
    """Upload every file under ``local_dir`` to ``out_uri``, preserving structure."""
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


def _ensure_single_process_group() -> None:
    """OLMo-core's DCP loader reads through ``torch.distributed.checkpoint``. Running under a
    plain ``python`` (no ``torchrun``), initialise a 1-rank gloo group so the load takes the
    standard single-process path rather than depending on a no-dist fast path."""
    import torch.distributed as dist  # noqa: PLC0415 -- lazy so --selftest stays torch-free

    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="gloo")


def _selftest() -> None:
    assert _split_uri("s3://b/a/c/") == ("b", "a/c/")
    assert _split_uri("s3://bucket/k") == ("bucket", "k")
    pfx = "teams/t/runs/r/checkpoints/dcp/"
    assert _rel_key(pfx + "config.json", pfx) == "config.json"
    assert _rel_key(pfx + "model_and_optim/__0_9.distcp", pfx) == "model_and_optim/__0_9.distcp"
    # A trailing-slash-free prefix argument must behave the same.
    assert _rel_key(pfx + "train/rank0.pt", pfx.rstrip("/")) == "train/rank0.pt"
    print("CONVERT SELFTEST OK: uri split + relative-key mapping verified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dcp-uri", help="s3:// prefix of the staged OLMo-core DCP checkpoint")
    ap.add_argument("--out", help="s3:// prefix to write the HF model to (EDULLM_CHECKPOINT_DIR)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HF tokenizer id to bundle")
    ap.add_argument("--dtype", default="bfloat16", help="saved weight dtype")
    ap.add_argument("--no-validate", action="store_true", help="skip logit-match validation")
    ap.add_argument("--selftest", action="store_true", help="offline helper checks and exit")
    ap.add_argument("--local-dcp", default="/tmp/base_dcp", help="local scratch for the DCP")
    ap.add_argument("--local-hf", default="/tmp/base_hf", help="local scratch for the HF model")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.dcp_uri or not args.dcp_uri.startswith("s3://"):
        print(f"--dcp-uri must be an s3:// prefix (got {args.dcp_uri!r})", file=sys.stderr)
        raise SystemExit(2)
    if not args.out or not args.out.startswith("s3://"):
        print(f"--out must be an s3:// prefix (got {args.out!r})", file=sys.stderr)
        raise SystemExit(2)

    # Everything below prints milestones to *stdout* (flushed). The prior run of this stage
    # exited 1 in ~6s with an empty ``edullm logs`` body: olmo_core's logging / an uncaught
    # traceback on stderr is not surfaced by the platform's log report, so the failure was
    # invisible. We therefore (a) echo a start banner + a milestone after each pre-download
    # step so the last line printed pinpoints where it died even with no traceback, and
    # (b) catch any exception and force its traceback onto stdout before re-raising.
    print(
        f"CONVERT START argv={sys.argv[1:]} cwd={os.getcwd()} python={sys.executable}",
        flush=True,
    )

    try:
        print("step: importing torch / olmo_core / edullm_data ...", flush=True)
        import torch  # noqa: PLC0415 -- lazy so --selftest stays offline
        from edullm_data.s3 import Boto3S3  # noqa: PLC0415
        from olmo_core.config import DType  # noqa: PLC0415
        from olmo_core.nn.hf import convert_checkpoint_to_hf, load_config  # noqa: PLC0415
        from olmo_core.utils import prepare_cli_environment  # noqa: PLC0415

        print("step: imports OK", flush=True)

        prepare_cli_environment()
        print("step: prepare_cli_environment OK", flush=True)
        _ensure_single_process_group()
        print("step: single-process group OK", flush=True)

        s3 = Boto3S3.default()
        print("step: Boto3S3.default() OK", flush=True)

        print(f"downloading DCP {args.dcp_uri} -> {args.local_dcp} ...", flush=True)
        n = download_prefix(s3, args.dcp_uri, args.local_dcp)
        print(f"downloaded {n} objects", flush=True)

        experiment_config = load_config(args.local_dcp)
        if experiment_config is None:
            raise RuntimeError("experiment config.json not found in the staged checkpoint")
        transformer_config_dict = experiment_config["model"]
        tokenizer_config_dict = experiment_config["dataset"]["tokenizer"]

        print("converting DCP -> HF (CPU) with logit validation ...", flush=True)
        convert_checkpoint_to_hf(
            original_checkpoint_path=args.local_dcp,
            output_path=args.local_hf,
            transformer_config_dict=transformer_config_dict,
            tokenizer_config_dict=tokenizer_config_dict,
            dtype=DType(args.dtype),
            tokenizer_id=args.tokenizer,
            validate=not args.no_validate,
            device=torch.device("cpu"),
        )

        print(f"uploading HF model -> {args.out} ...", flush=True)
        uploaded = upload_dir(s3, args.local_hf, args.out)
        for u in uploaded:
            print(f"  wrote {u}", flush=True)

        print(f"CONVERT OK: hf_base={args.out.rstrip('/')} files={len(uploaded)}", flush=True)
    except BaseException:
        import traceback as _tb  # noqa: PLC0415

        print("CONVERT FAILED -- traceback (forced to stdout):", flush=True)
        _tb.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    main()
