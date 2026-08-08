"""Stages 3-4: train the ORM / PRM reward models on PRM800K *scores* (not preference pairs).

``open_instruct/reward_modeling.py`` is a Bradley-Terry trainer: it needs (chosen, rejected)
pairs and maximises a sigmoid margin. PRM800K gives **pointwise labels** — a whole-solution
correctness bit (ORM) and per-step ``-1/0/+1`` ratings (PRM) — so this is a pointwise fork of
that trainer sharing its backbone/head/pooling but swapping the paired loss + dataloader:

  * ``--rm-type orm``: ``num_labels=1``, BCE-with-logits on the last-token score against the
    0/1 outcome (Cobbe-style whole-solution verifier).
  * ``--rm-type prm``: ``num_labels=3``, cross-entropy on the last-token score against the step
    rating class {neg=0, neu=1, pos=2}. Each ``(prefix, step)`` example is one sequence ending
    at the step boundary, so last-token pooling reads the right position with no post-
    tokenization index alignment (see ``rm_common``).

Both warm-start from the **SFT model** (Stage 2), share tokenizer + tulu formatting with SFT and
GRPO, and save as a full HF ``*ForSequenceClassification`` model so the Stage-6 verifier loads
either RM through one code path.

This is the ``accelerate launch`` target; ``run_rm.py`` does the S3 download/upload around it.
Run (per RM) as an eduLLM ``-train`` GPU job (W&B auto-wired by the GPU role)::

    accelerate launch --mixed_precision bf16 --num_processes N \\
        projects/prm_vs_orm/reward_modeling_scored.py --rm-type {orm,prm} \\
        --model-name-or-path <sft_dir> --data-file <orm|prm.jsonl> --output-dir <out> \\
        --with-tracking

``--selftest`` runs the offline tokenization/label/collator checks (no torch weights, no S3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_common  # noqa: E402 -- overlay-local module, added to path above

# accelerate/torch.distributed.elastic swallows the failing rank's traceback and prints only a
# per-rank exit-code summary, so the real exception never reaches the ~50-line tail `edullm logs`
# returns. We re-emit it line-by-line with a greppable prefix that run_rm.py pulls out and prints
# LAST (same self-diagnosing pattern that unblocked the SFT stage; see _sft_launch.py).
_ERR_MARKER = "RMERR| "


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_examples(
    tokenizer: Any, rows: list[dict], rm_type: str, *, add_bos: bool, max_length: int
) -> list[tuple[list[int], Any]]:
    """Tokenize every row to (ids, label). ORM labels are floats, PRM labels are class ints."""
    out: list[tuple[list[int], Any]] = []
    for row in rows:
        if rm_type == "orm":
            out.append(rm_common.orm_example_ids(tokenizer, row, add_bos=add_bos, max_length=max_length))
        else:
            out.append(rm_common.prm_example_ids(tokenizer, row, add_bos=add_bos, max_length=max_length))
    return out


def _detect_add_bos(tokenizer: Any, choice: str) -> bool:
    if choice == "yes":
        return True
    if choice == "no":
        return False
    return tokenizer.bos_token_id is not None  # "auto"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rm-type", choices=["orm", "prm"], help="reward-model kind")
    ap.add_argument("--model-name-or-path", help="local SFT model dir to warm-start from")
    ap.add_argument("--tokenizer-name-or-path", default=None, help="defaults to --model-name-or-path")
    ap.add_argument("--data-file", help="local orm.jsonl / prm.jsonl")
    ap.add_argument("--output-dir", help="local dir to save the trained RM to")
    ap.add_argument("--add-bos", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--num-train-epochs", type=float, default=1.0)
    ap.add_argument("--per-device-batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lr-scheduler-type", default="linear")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--max-examples", type=int, default=0, help="cap rows (0 = all); for smoke runs")
    ap.add_argument("--dtype", default="bfloat16", help="named so the precision guard can see it")
    ap.add_argument("--with-tracking", action="store_true")
    ap.add_argument("--wandb-project-name", default=None, help="defaults to prm-vs-orm-<rm_type>")
    ap.add_argument("--wandb-entity", default="eduLLM")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rm_common._selftest()
        print("REWARD_MODELING_SCORED SELFTEST OK")
        return

    for name, val in (("--rm-type", args.rm_type), ("--model-name-or-path", args.model_name_or_path),
                      ("--data-file", args.data_file), ("--output-dir", args.output_dir)):
        if not val:
            print(f"{name} is required", file=sys.stderr)
            raise SystemExit(2)

    import numpy as np  # noqa: PLC0415
    import olmo3_adapter  # noqa: PLC0415 -- overlay-local (path inserted above)
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415
    from accelerate import Accelerator  # noqa: PLC0415
    from accelerate.utils import set_seed  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler  # noqa: PLC0415

    from open_instruct.model_utils import disable_dropout_in_model, save_with_accelerate  # noqa: PLC0415

    olmo3_adapter.register()

    num_labels = 1 if args.rm_type == "orm" else rm_common.NUM_PRM_CLASSES
    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token_id is None:
        # explicit-mask pooling tolerates pad==eos; just need *a* pad id to fill with.
        tokenizer.pad_token = tokenizer.eos_token
    add_bos = _detect_add_bos(tokenizer, args.add_bos)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)

    rows = load_jsonl(args.data_file)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    examples = build_examples(
        tokenizer, rows, args.rm_type, add_bos=add_bos, max_length=args.max_seq_length
    )
    if accelerator.is_main_process:
        print(
            f"[{args.rm_type}] {len(examples)} examples | num_labels={num_labels} | add_bos={add_bos} "
            f"| pad_id={tokenizer.pad_token_id} eos_id={tokenizer.eos_token_id}",
            flush=True,
        )

    collator = rm_common.ScoredCollator(
        pad_token_id=tokenizer.pad_token_id, label_dtype="float" if args.rm_type == "orm" else "long"
    )
    dataloader = DataLoader(
        examples, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collator
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, torch_dtype=torch.bfloat16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    embeddings = model.get_input_embeddings()
    if len(tokenizer) > embeddings.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    disable_dropout_in_model(model)
    # fresh classification head init (p.11 of https://arxiv.org/abs/2009.01325)
    torch.nn.init.normal_(model.score.weight, std=1 / np.sqrt(model.config.hidden_size + 1))

    steps_per_epoch = max(1, len(dataloader) // args.gradient_accumulation_steps)
    num_training_steps = int(steps_per_epoch * args.num_train_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    if args.with_tracking and accelerator.is_main_process:
        import wandb  # noqa: PLC0415

        wandb.init(
            project=args.wandb_project_name or f"prm-vs-orm-{args.rm_type}",
            entity=args.wandb_entity,
            name=f"{args.rm_type}-rm",
            config=vars(args),
        )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    global_step = 0
    for epoch in range(max(1, round(args.num_train_epochs))):
        for batch in dataloader:
            with accelerator.accumulate(model):
                pooled = rm_common.pooled_scores(model, batch["input_ids"], batch["attention_mask"])
                labels = batch["labels"]
                if args.rm_type == "orm":
                    logit = pooled.squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logit, labels.to(logit.dtype))
                    acc = ((logit > 0).long() == labels.long()).float().mean()
                else:
                    loss = F.cross_entropy(pooled, labels)
                    acc = (pooled.argmax(-1) == labels).float().mean()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    m = accelerator.reduce(
                        torch.stack([loss.detach(), acc.detach()]), reduction="mean"
                    ).tolist()
                    if accelerator.is_main_process:
                        print(
                            f"epoch {epoch} step {global_step}/{num_training_steps} "
                            f"loss {m[0]:.4f} acc {m[1]:.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                            flush=True,
                        )
                        if args.with_tracking:
                            import wandb  # noqa: PLC0415

                            wandb.log(
                                {
                                    f"train/{args.rm_type}/loss": m[0],
                                    f"train/{args.rm_type}/acc": m[1],
                                    f"train/{args.rm_type}/lr": scheduler.get_last_lr()[0],
                                },
                                step=global_step,
                            )

    os.makedirs(args.output_dir, exist_ok=True)
    save_with_accelerate(
        accelerator, model, tokenizer, args.output_dir, chat_template_name="tulu"
    )
    if accelerator.is_main_process:
        print(f"RM OK: rm_type={args.rm_type} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # re-emit the full traceback with a greppable per-line marker, then re-raise so the
        # process still exits non-zero. Only the failing rank(s) reach here.
        _tb = traceback.format_exc()
        sys.stderr.write("".join(f"{_ERR_MARKER}{ln}\n" for ln in _tb.splitlines()))
        sys.stderr.flush()
        raise
