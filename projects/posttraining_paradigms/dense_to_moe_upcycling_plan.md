# Dense-to-MoE upcycling implementation plan

## Decision and scope

This plan converts the teacher, `Qwen/Qwen2.5-3B-Instruct`, into a true sparse
feed-forward mixture of experts without repeating base-model pretraining.

This is a research control, not the recommended first implementation for the
teacher project. Dense-to-MoE upcycling usually needs far more continued
pretraining data than the 455 ESTELA questions. Its purpose here is to answer a
narrow engineering question:

> Can a function-preserving sparse expansion of the teacher acquire useful
> pedagogical-strategy specialization under a realistic continued-training
> budget?

The first milestone replaces only four late transformer MLPs. Expanding all
layers is allowed only after the partial conversion passes equivalence,
optimization, and specialization gates.

## Architecture

### Dense block

Qwen's SwiGLU MLP computes:

```text
y = down_proj(silu(gate_proj(x)) * up_proj(x))
```

### Upcycled block

Replace selected MLPs with:

```text
router_logits = router(x)
selected = top_k(router_logits, k=2)
y = sum(normalized_router_weight_i * expert_i(x))
```

Initial configuration:

- experts per converted layer: 4;
- active experts per token: 2;
- converted layers: the final 4 transformer layers;
- attention, embeddings, norms, and language head remain dense;
- each expert begins as an exact clone of the original MLP;
- router weights use a small zero-mean Gaussian initialization; and
- selected router weights are renormalized to sum to one.

Because every expert initially computes the same function, any normalized
mixture reproduces the original dense MLP. Top-2 is used for the first version
because top-1 identical experts provide no language-model gradient to the
router before expert divergence.

### Router losses

Train with:

```text
loss =
  causal_lm_loss
  + 0.01 * load_balance_loss
  + 0.001 * router_z_loss
  + distillation_weight * dense_output_kl
```

The load-balancing coefficient is a starting value, not a constant to preserve
at all costs. Report both routing balance and task loss because uniform routing
can suppress real specialization.

## Repository layout

Create:

```text
open-instruct/projects/dense_to_moe_upcycling/
  __init__.py
  upcycled_mlp.py
  upcycled_qwen.py
  convert_checkpoint.py
  build_cpt_mixture.py
  train_cpt.py
  evaluate.py
  inspect_routing.py
  tests/
    test_function_equivalence.py
    test_sparse_dispatch.py
    test_checkpoint_round_trip.py
    test_router_losses.py
  scripts/
    convert_smoke.sh
    train_partial_moe.sbatch
    train_full_moe.sbatch
    evaluate_moe.sbatch
```

Do not modify the installed Transformers Qwen implementation. The project
wrapper should:

1. load the ordinary dense checkpoint;
2. replace specified `model.layers[i].mlp` modules in memory;
3. save a conversion manifest plus the sparse-module state; and
4. reconstruct the model from the base checkpoint and manifest.

This avoids publishing a custom `model_type` before the architecture is
validated.

## Phase 1: function-preserving conversion

Implement `UpcycledQwenMLP` with:

- a `ModuleList` of cloned Qwen MLP experts;
- a bias-free linear router from hidden width to expert count;
- top-k routing and normalized selected weights;
- token-to-expert dispatch that preserves token order;
- expert usage counters; and
- optional dense reference execution for tests.

The initial implementation may loop over experts. It is a correctness
prototype, not a throughput claim. Add grouped GEMM or a sparse-MoE runtime only
after numerical and behavioral equivalence pass.

Required tests:

- converted and dense MLP outputs differ by at most `1e-5` in float32;
- full-model next-token logits differ by at most `1e-4` in float32;
- every token is dispatched to exactly two experts;
- selected weights sum to one per token;
- gradients reach selected experts and the auxiliary router losses;
- saving and restoring preserves logits; and
- dense layers not listed in the manifest remain byte-identical.

## Phase 2: data construction

ESTELA is evaluation and pedagogical-control data, not a continued-pretraining
corpus. Build a separate mixture containing:

- 70% general text and instruction data;
- 20% science and physics explanations/problems; and
- 10% educational dialogue.

Deduplicate against all sealed ESTELA prompts and solutions. Store source,
license, document ID, and contamination hashes.

Budgets:

1. plumbing smoke: 1 million tokens;
2. partial-MoE pilot: 10–50 million tokens;
3. full conversion consideration: at least 100 million tokens; and
4. serious upcycling claim: hundreds of millions to billions of tokens.

The pilot can determine whether the implementation trains; it cannot establish
the pretraining-scale benefit reported by sparse-upcycling papers.

## Phase 3: optimization schedule

### Stage A: equivalence and router warm-up

- Load the converted model from the dense checkpoint.
- Freeze experts and all dense weights.
- Train only router parameters for 500–2,000 steps using load-balance and
  z-loss.
- Confirm all experts receive tokens.

The causal language loss cannot distinguish identical experts at this stage.
Router warm-up is only for healthy dispatch.

### Stage B: controlled expert divergence

- Unfreeze expert MLPs and routers in converted layers.
- Keep attention and unconverted MLPs frozen for the first pilot.
- Train with causal loss and KL to frozen dense logits.
- Use learning rates around `1e-4` for routers and `1e-5` for experts.
- Apply expert dropout only if experts remain highly correlated.
- Do not partially reinitialize weights until exact-clone training has been
  measured.

### Stage C: optional diversification

If cloned experts remain functionally identical:

- branch experts on balanced domain subsets for a short warm-up; or
- reinitialize 5–10% of matched gate/up columns and down rows per expert.

Large Drop-Upcycling resets such as 50% are inappropriate for the initial
small-data experiment.

### Stage D: full-layer conversion

Convert all MLP layers only if the four-layer model:

- preserves the dense model;
- trains without dead experts;
- develops repeatable routing specialization; and
- improves a held-out metric over dense continuation.

## Phase 4: evaluation

### Required baselines

Use identical tokens, optimization steps, and seeds:

1. frozen dense checkpoint;
2. dense continued pretraining;
3. dense LoRA continued training;
4. cloned MoE with frozen experts;
5. trained partial MoE; and
6. a total-parameter-matched dense adapter where feasible.

Report both equal-active-compute and equal-total-parameter comparisons. A
sparse model has more resident parameters even when it activates fewer.

### Metrics

- validation perplexity and next-token KL to the original model;
- general and physics benchmark accuracy;
- ESTELA accuracy by bank and question type;
- worst-bank performance;
- expert load, entropy, dead-expert count, and overflow;
- pairwise expert-output cosine similarity;
- mutual information between source/domain and routing;
- peak memory, batch-1 latency, and throughput; and
- unrelated-capability retention.

### Pedagogical-expert test

After generic CPT, condition prompts on held-out mastery states and pedagogical
actions such as diagnostic questioning, contrast cases, worked examples, and
retrieval practice. Test whether routing changes predictably and whether the
resulting intervention improves held-out partner performance. The synthetic
student never receives router IDs. Action-correlated routing is not sufficient:
the selected expert must causally improve the downstream outcome.

## Stop/go gates

Stop before substantial CPT if:

- conversion is not function preserving;
- checkpoint restoration changes logits;
- any expert receives under 5% of tokens after router warm-up; or
- memory or wall-clock cost exceeds the dense baseline by more than expected
  from active expert count.

Stop the upcycling hypothesis after the partial pilot if:

- dense continuation is better under the same token budget;
- experts remain nearly identical across three seeds;
- routing follows lexical templates but not held-out domains;
- general retention drops more than 2 percentage points; or
- gains disappear when bootstrapping by bank/domain.

Proceed to full conversion only if partial MoE beats dense continuation on
held-out loss and at least one downstream metric without a retention loss.

## Compute estimate

The four-layer 3B prototype should fit on one H100 and may fit on an L40S with
activation checkpointing and frozen dense weights. Full E=4, top-2 expansion
substantially increases resident parameters and optimizer state; plan for
multiple GPUs or ZeRO/FSDP rather than assuming a single H100. Inactive experts
still consume memory.

Approximate pilot:

- conversion and tests: CPU or one GPU, under one hour;
- 10–50M-token partial CPT: roughly one H100-day, depending on sequence length;
- 100M-token full pilot: multiple GPU-days, highly dependent on sharding and
  the naive-versus-grouped dispatch implementation.

This plan should not be launched merely because it is technically feasible.
The routed teacher-LoRA plan is the lower-cost falsification of whether
pedagogical specialization helps this project at all.

## Partial implementation result

A minimal true FFN-MoE path is now implemented in
`projects/dense_to_moe_upcycling/`. This does not supersede the scientific
gates above; it establishes that the architecture and training path work.

The completed smoke:

- sampled 1,006,084 Qwen tokens from 1,732 documents across eight semantic
  clusters in the community raw-text `gvlassis/ClimbMix` release;
- replaced Qwen 2.5 3B teacher layers 34 and 35 with four cloned SwiGLU
  experts and top-2 learned routing;
- froze the remaining dense model and optimized 541.1M router/expert
  parameters for 50 steps, or 51,200 tokens, on one H100;
- produced a 3.492B-parameter resident model;
- retained close bfloat16 initial behavior: dense/upcycled loss
  2.67008/2.66952 and mean absolute logit difference 0.00536;
- moved held-out loss from 2.74258 to 2.74236, which is effectively flat at
  this smoke-test scale;
- obtained balanced expert loads of 23.4–26.8% in layer 34 and 22.2–28.8% in
  layer 35, with no dead experts; and
- saved, restored, and generated coherently from the MoE checkpoint.

The checkpoint is stored on ORCD under
`projects/dense_to_moe_upcycling/runs/partial_moe/moe_checkpoint.pt`.

One implementation issue was falsified and fixed during the run. Collecting
router auxiliary losses through module attributes while using gradient
checkpointing detached those losses from the active backward graph. The
successful run disabled checkpointing and directly verified nonzero router
gradient norms. A scale-up must carry auxiliary losses through the checkpointed
forward or use a tested non-reentrant path.

This is a plumbing success, not a model-quality result. The next gate remains a
matched dense continuation versus partial MoE over at least 10M tokens, with
loss, retention, throughput, expert divergence, and domain-routing evaluation.
