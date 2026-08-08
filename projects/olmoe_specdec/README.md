# Speculative decoding for the OLMoE RLVR rollout

Does the acceleration in [arXiv:2604.26779](https://arxiv.org/abs/2604.26779) — speculative
decoding as a *lossless* rollout accelerator inside an RL loop, 1.5–1.8× generation and up to
1.41× end-to-end at 8B — reproduce on the final RLVR step that turns
`allenai/OLMoE-1B-7B-0125-DPO` into `allenai/OLMoE-1B-7B-0125-Instruct`?

**Deliverable: the speedup measurement and a full-recipe projection.** Not the Instruct
checkpoint. The RLVR step is the measurement workload here, and runs are short.

**Losslessness is not something we are testing.** Rejection sampling makes the accepted tokens
the target policy's own samples, so this changes rollout throughput and nothing about what is
sampled or optimised. It is verified once (step 1c below) and then relied on.

## Why this may not reproduce, and why that decides the order of work

The paper's gain comes from a **dense 8B** model emitting **long reasoning traces**, where
generation is 65–72% of step time. This workload differs on every axis: OLMoE is **1B-active
MoE**, RLVR-GSM answers are **short**, and RL rollout batches are **large** (48 prompts × 16
samples = 768 concurrent sequences at the documented settings). Large-batch MoE decode activates
most experts and trends compute-bound, which is where verification overhead is hardest to
amortise.

The paper's own Table 2 has a drafting method with a *positive* acceptance length of 2.47 running
at **0.7× — slower than baseline**. Positive acceptance is not sufficient.

So the work is ordered cheapest-decisive-first, and **a documented null is an acceptable
outcome.**

## Order of GPU work

### 1a. Autoregressive baseline generation — needs no new code, do this first

`open_instruct/benchmark_generators.py` already reports `avg_tokens_per_second`,
`generation_time_percentage`, MFU and per-response lengths. Run it on the DPO checkpoint with no
draft, sweeping batch size up to the real 768. This prices the **Amdahl ceiling** before a draft
model is ever trained.

### 1b. R_gen from a short RL run

~20 steps of `grpo_fast.py` with `--vllm_collect_spec_decode_stats`. Logs
`spec/generation_share` directly. This is the R_gen in the bound below; there is no substitute
for measuring it, and it is the number most likely to differ from the paper.

### 1c. Losslessness check

Greedy-decode (`--temperature 0`) a fixed prompt set through an autoregressive engine and an
EAGLE-3 engine and diff the token ids. They must be identical. Cheap, and it is the premise
everything else rests on.

### 2. On-policy corpus for draft initialisation

```bash
python projects/olmoe_specdec/generate_rollout_corpus.py \
    --model_name_or_path allenai/OLMoE-1B-7B-0125-DPO \
    --dtype bfloat16 \
    --output_path "$EDULLM_OUTPUT_PREFIX/rollout_corpus.jsonl"
```

Table 3 is why this is not "some text": at matched k=3, a chat-data draft got 1.51× and a draft
trained on the actual post-training prompts got 1.77×. The script mirrors the RLVR sampling
settings and takes its tokenizer and chat template from open-instruct's own `TokenizerConfig`, so
the draft is fitted to the distribution the rollout actually produces. It writes a manifest
beside the corpus recording every setting, so a draft can be traced to what it was fitted to.

Add `--max_prompts 32` for a smoke test.

### 3. Train the EAGLE-3 draft

With [`vllm-project/speculators`](https://github.com/vllm-project/speculators) — vLLM-native, so
the thing we train against is the thing we serve with. Target geometry: `hidden_size 2048`,
`vocab_size 50304`, `tie_word_embeddings false`, 16 layers.

Two design points that are not free choices:

- **Keep the draft dense, not MoE.** SpecForge's result is that at low top-k each expert sees too
  few tokens and generalises worse, while raising top-k raises per-token drafting FLOPs — MoE
  works as a *target* and not as a *draft*.
- **Use EAGLE-3's reduced draft vocab** (`t2d`/`d2t`). A full 50304-row `lm_head` at hidden 2048
  is ~103M parameters, larger than the draft's transformer layer, and drafting cost is exactly
  what erases speedup as k grows (Table 4).

There is no pretrained EAGLE-3 draft for OLMoE to fall back on.

### 4. The gate

Sweep `k ∈ {1, 3, 5}` × batch size × response length, autoregressive vs EAGLE-3, through
`benchmark_generators.py`. Then evaluate the paper's §2.2 bound at the measured values:

```
S_step <= 1 / (R_gen / α + (1 - R_gen))
```

`open_instruct/spec_decode/reporting.py` computes this per step and logs it beside the realised
step time. **Require a measured generation speedup ≥ 1.25× at batch 768.** Below that, stop and
write up the null rather than paying for paired RL runs.

Reading the bound matters as much as the speedup: if the measured number tracks the bound, the
integration works and the workload simply has little generation to accelerate. If it falls well
below, the loss is overhead — drafting cost, verification, batch effects. Those lead to opposite
decisions, and "we got 1.05×" alone cannot distinguish them.

### 5. Paired RL runs — only if the gate clears

Two `grpo_fast.py` arms at matched seed, differing **only** in `speculative_config`.

## Flags

| flag | notes |
|---|---|
| `--vllm_speculative_method eagle3` | only `eagle3` is accepted; see below on n-gram |
| `--vllm_speculative_model <path>` | required with a method; no pretrained draft exists |
| `--vllm_num_speculative_tokens 3` | k. Default 3, and Table 4 says higher is *worse* end-to-end |
| `--vllm_speculative_draft_tensor_parallel_size` | omit to let vLLM default it |
| `--vllm_collect_spec_decode_stats` | **set on both arms** — see below |

All default to off; with the method unset the engine is built with `speculative_config=None`,
which is `AsyncEngineArgs`' own default.

**`--vllm_collect_spec_decode_stats` goes on both arms.** Enabling stats costs a little time per
engine iteration, and the deliverable is a *ratio of step times* between two arms — anything paid
in one arm only lands directly in the result. In the baseline arm it records nothing (`num_drafts`
stays 0 and α is reported absent, not 1.0) but it still costs the same, so it cancels.

**`--use_vllm_logprobs` must stay off.** With it on, the loss consumes logprobs returned by an
engine that is speculating. Leaving it off forces learner-side recomputation and makes
verifier-exactness structural rather than an assumption about how vLLM reports logprobs.

**n-gram drafting is rejected, not merely unimplemented.** The paper measures it at 0.7× on
RL-Zero and 0.5× on RL-Think — slower than no speculation — despite acceptance lengths of 2.47
and 2.05.

## Keeping the arms comparable

- Same seed, same `--pack_length 4096`, same batch shape. Regrouping micro-batches changes each
  example's contribution to the loss.
- 8×A100 on this account is **40 GB per card, not 80** — 320 GB total. OLMoE is 14 GB in bf16, and
  AdamW fp32 moments, a reference policy and the vLLM engine all have to fit alongside.
- Name `bfloat16` in the submitted command text. The platform's `bfloat16_not_in_the_hardware`
  guard reads the command as text and cannot see a precision set in code.

## Submitting

Start with `edullm check --json` and read `cost` and `approval_class` out of **its** output.
Do not quote a price, a runtime bound or an approval tier from this file or any other — those live
in reviewed configuration that changes without notice.

The image builds only from a push to a branch named `edullm/**` (or `main`), takes 3–8 minutes,
and the build gate runs
`ruff==0.14.13 check --no-cache --config pyproject.toml open_instruct projects mason.py`.

## Layout

| path | what |
|---|---|
| `open_instruct/spec_decode/` | the reusable layer: config, EAGLE-3 target, registration, metrics, reporting |
| `projects/olmoe_specdec/` | this experiment's scripts and runbook |

`PATCHES.md` documents every upstream file touched and the rebase risk of each.
