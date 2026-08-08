# Dense-to-MoE teacher upcycling

This directory contains a working partial conversion of
`Qwen/Qwen2.5-3B-Instruct` from dense SwiGLU blocks to sparse top-k expert
blocks. It is an infrastructure and optimization smoke test, not yet evidence
that an MoE teacher is better than its dense parent.

## Implemented path

1. `sample_climbmix.py` streams a small raw-text sample from eight semantic
   clusters in `gvlassis/ClimbMix`.
2. `upcycled_mlp.py` clones a dense Qwen MLP into identical experts, installs a
   learned router, dispatches tokens with top-k routing, computes load-balancing
   and router z-losses, and saves/restores expert-only checkpoints.
3. `train_cpt.py` freezes the dense backbone and continue-trains the routers and
   experts.
4. `evaluate_checkpoint.py` restores the saved expert checkpoint over the dense
   base and runs generation.
5. `test_upcycled_mlp.py` tests initial equivalence, router gradients, layer
   selection, and checkpoint round trips on a tiny Qwen model.

Generated data, logs, and checkpoints live under `runs/` and are ignored by
git.

## Completed 3B smoke

The GPU run converted layers 34 and 35 to four experts with top-2 routing:

- data: 1,006,084 Qwen tokens, 1,732 documents, eight ClimbMix clusters;
- resident parameters: 3.492B;
- trainable router and expert parameters: 541.1M;
- optimization: 50 steps, 51,200 tokens, one H100;
- initial mean absolute logit difference from dense: 0.00536 in bfloat16;
- initial dense/upcycled loss: 2.67008 / 2.66952;
- held-out loss before/after the smoke: 2.74258 / 2.74236;
- layer 34 expert load: 24.6%, 26.8%, 25.3%, 23.4%;
- layer 35 expert load: 28.8%, 22.2%, 24.1%, 25.0%; and
- dead experts: zero.

The checkpoint restored successfully and generated a coherent tutoring turn.
The saved checkpoint is on ORCD at:

```text
/home/zsophia/orcd/scratch/open-instruct/projects/dense_to_moe_upcycling/runs/partial_moe/moe_checkpoint.pt
```

The tiny local tests pass with:

```bash
python -m pytest projects/dense_to_moe_upcycling/test_upcycled_mlp.py -q
```

## Reproduce

Sample data:

```bash
python -m projects.dense_to_moe_upcycling.sample_climbmix \
  --out projects/dense_to_moe_upcycling/runs/data/climbmix_1m.jsonl \
  --target-tokens 1000000
```

Train a partial MoE:

```bash
python -m projects.dense_to_moe_upcycling.train_cpt \
  --data projects/dense_to_moe_upcycling/runs/data/climbmix_1m.jsonl \
  --out-dir projects/dense_to_moe_upcycling/runs/partial_moe \
  --layers last2 --num-experts 4 --top-k 2
```

For the actual H100 configuration, use `train_partial_moe.sbatch`.

## Interpretation and next gate

This run proves conversion, sparse dispatch, router optimization, checkpoint
restoration, and generation. The 50-step loss change is too small and too
underpowered to establish a quality gain. The next valid experiment is a
matched dense-continuation control over at least 10M tokens, followed by
held-out loss, general-retention, throughput, expert-divergence, and
domain-routing measurements.

Do not enable the current external router auxiliary loss together with
reentrant gradient checkpointing: the stored router tensors may be detached
from the backward graph. The successful run leaves checkpointing disabled. A
larger run should return auxiliary losses through the checkpointed forward or
use a verified non-reentrant implementation.

The sampled community ClimbMix release is CC BY-NC 4.0 and is suitable for this
non-commercial research experiment.
