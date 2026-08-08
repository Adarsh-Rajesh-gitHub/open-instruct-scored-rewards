# OLMoE-1B-7B-0125 post-training: the DPO leg, and two MoE flag decisions

**From:** Adarsh Rajesh
**To:** Stephen Zhang (`syz2026`)
**Covers:** Aug 8, 2026
**Read this if:** you are picking up stage 3 of the SFT+DPO smoke test on `allenai/OLMoE-1B-7B-0125`,
or either of the two MoE-specific flag questions that came out of building it.

---

## Orientation (read first)

Meric asked three of us to smoke-test OLMo's default SFT+DPO on OLMoE-1B-7B-0125 by end of day.
"Default SFT+DPO" means two programs in this repository: `open_instruct/finetune.py` and
`open_instruct/dpo_tune_cache.py`. Sid owns the RL leg separately and it is not in scope here.

**I own the SFT leg.** Stage 1 of the run spec is mine: the LoRA fine-tune through `finetune.py`,
its flags, its rank choice, and defending the parameter arithmetic behind it. **You own stage 3,
the DPO leg**, which consumes the checkpoint stage 1 produces, plus the two flag decisions below
that neither of us can answer from the smoke test alone.

**Bottom line: the whole three-stage run is written, priced and accepted by the platform, and
nothing has executed.** `edullm check` returns `refused: false`. What stands between us and a
result is one push permission, not any remaining engineering.

### Blocker for you specifically

**Neither of us can push to an `edu-llm` fork, and you are not a faster route than I am.** I
checked both of us against the roster rather than assuming:

```bash
gh api repos/edu-llm/open-instruct --jq .permissions
# {"admin":false,"maintain":false,"pull":true,"push":false,"triage":false}
gh api repos/edu-llm/open-instruct-scored-rewards --jq .permissions
# same
```

`config/organization.yaml` on `edu-llm/platform` puts you and me both in `memory-split` and
neither of us in `post-training`. No GitHub team grants access to either open-instruct repo at
all — `orgs/edu-llm/teams/<team>/repos` returns only `platform` for `post-training`,
`memory-split` and `platform`, and `OLMo-core` plus `platform` for `team-members`. So push on
these two repos is a direct collaborator grant, and team membership will not get either of us one.

**The fastest unblock is Sid.** He has demonstrably pushed an `edullm/**` branch to the registered
fork:

```bash
gh api repos/edu-llm/open-instruct-scored-rewards/commits/$(gh api \
  repos/edu-llm/open-instruct-scored-rewards/branches/edullm%2Folmoe-specdec --jq .commit.sha) \
  --jq .author.login
# sidvenkatayogi
```

Sophia Zhang (`zsophiaaa`), Tom Liu (`pianomaster99`) and Frank (`philote-dev`) have each pushed
one too. Any of them can push my branch and the run goes. Access is also filed as
[`edu-llm/platform#435`](https://github.com/edu-llm/platform/issues/435), but that is the slow path.

---

## Current state

| | |
| --- | --- |
| Branch | `edullm/olmoe-sft-dpo-smoke`, 1 commit `8e2ea2a` on top of `origin/main` |
| Pushed to | `https://github.com/Adarsh-Rajesh-gitHub/open-instruct-scored-rewards` (my fork) |
| Pushed to `edu-llm` | **No.** This is the blocker. |
| `edullm check` | `refused: false`, approval `automatic`, `gpu-1xl40s`, 1 node, ceiling **$5.58** at $1.861/hr over 3h |
| Billed to | team `memory-split`, `wandb_project: memory-split` — not post-training. See the Slack note. |
| Stages executed | **None.** No GPU has run any of this. |

Get it locally with:

```bash
git clone https://github.com/Adarsh-Rajesh-gitHub/open-instruct-scored-rewards.git
cd open-instruct-scored-rewards && git checkout edullm/olmoe-sft-dpo-smoke
```

### Files

Commit `8e2ea2a` adds exactly two files, 217 lines, and touches nothing else:

- `.edullm/run.yaml` (88) — the three-stage command, and a comment block giving the reason for
  every non-obvious choice in it. This is the file the platform reads.
- `.edullm/validate_smoke_flags.py` (129) — an `ast`-based checker that reads the trainers'
  dataclasses without importing them and asserts every `--flag` in `run.yaml` resolves to a parsed
  field. Runs on a laptop with no torch.

```bash
python3 .edullm/validate_smoke_flags.py
# checked 59 flags across 3 programs
# every flag resolves to a parsed field
```

**Run that before you change any flag.** It is the only thing standing between a typo and a paid
machine dying at argument-parse time.

### The three stages

All in `.edullm/run.yaml`, chained with `&&`, on one L40S:

1. **SFT** — `open_instruct/finetune.py`, LoRA r=16, `allenai/OLMoE-1B-7B-0125`,
   `allenai/tulu-3-sft-personas-algebra`, 64 samples, seq len 1024 → `/tmp/olmoe-smoke/sft`. Mine.
2. **Merge** — `open_instruct/merge_lora.py` folds the adapter into the base weights →
   `/tmp/olmoe-smoke/sft_merged`. Mine, but it is the seam between us: stage 3 reads this path.
3. **DPO** — `open_instruct/dpo_tune_cache.py`, LoRA r=16 on the merged checkpoint,
   `allenai/tulu-3-wildchat-reused-on-policy-8b`, 32 samples, `--loss_type dpo_norm --beta 5`,
   lr 5e-7 → `/tmp/olmoe-smoke/dpo`. **Yours.**

Stage 3 takes `--model_name_or_path "$OUT/sft_merged"`, so DPO trains a *fresh* adapter on a dense
merged checkpoint rather than stacking a second adapter on the first. That is deliberate and it is
what the merge stage exists for.

---

## What you own

### 1. The DPO leg (stage 3)

Three things in it I could not settle without a GPU, in the order they will bite:

**The archived recipe's DPO flags are five renames out of date, not three.** The commit message on
`8e2ea2a` says three. It undercounts. `docs/archived_dev_scripts/olmoe_0125.sh` is AI2's own
OLMoE-0125 recipe and every DPO invocation in it uses names `dpo_tune_cache.py` no longer parses,
because its config moved onto the OLMo-core `TrainingConfig`:

| Archived recipe | What `DPOExperimentConfig` parses today |
| --- | --- |
| `--dpo_loss_type` | `--loss_type` |
| `--dpo_beta` | `--beta` |
| `--num_train_epochs` | `--num_epochs` |
| `--dataset_mixer_list` | `--mixer_list` |
| `--gradient_checkpointing` | *gone* — see below |

`finetune.py` still parses `--num_train_epochs` and `--dataset_mixer_list`, so the two trainers
genuinely disagree on those names. Do not copy a flag from the SFT stage into the DPO stage.
Reproduce the table with:

```bash
python3 -c "
import sys; sys.path.insert(0,'.edullm')
from validate_smoke_flags import fields_of, SHARED, DPO_MODULES
d=fields_of('open_instruct/dpo_utils.py','DPOExperimentConfig',DPO_MODULES)
s=fields_of('open_instruct/finetune.py','FlatArguments',SHARED)
for n in ['loss_type','beta','num_epochs','num_train_epochs','mixer_list','dataset_mixer_list']:
    print(f'{n:22} DPO={n in d!s:5} SFT={n in s}')"
```

**Stage 3 runs with no activation checkpointing and that is the likeliest place it dies.** SFT
takes `--gradient_checkpointing`; DPO has no such flag. Its equivalent is
`activation_memory_budget` (`open_instruct/olmo_core_utils.py:107`, default `1.0`), and
`dpo_tune_cache.py:363` and `:377` only enable checkpointing when it is `< 1`. The run spec leaves
it at the default, so stage 3 has checkpointing off on a 48 GB card while holding a merged 6.9B
base plus a fresh adapter plus DPO's paired chosen/rejected forward passes. If it OOMs, the first
thing to try is `--activation_memory_budget 0.5`, not a smaller batch.

**The reference logprobs are cached with the LoRA adapter attached, not disabled.**
`dpo_tune_cache.py:533` passes `disable_adapter_context=None` into
`dpo_utils.build_reference_logprobs_cache`, and that function only enters the disable-adapter
branch when `use_lora and disable_adapter_context is not None` (`dpo_utils.py:463`). For our run
this is harmless: the cache is built before the training loop and PEFT zero-initialises `lora_B`,
so a fresh adapter is an exact identity and the reference equals the base model. It stops being
harmless the moment anyone resumes from a trained adapter or reuses a stale cache file — the cache
is keyed by `compute_reference_cache_hash(args, tc)` and loaded straight off disk if the path
exists (`dpo_utils.py:436`). Worth a comment in the code at minimum; worth a bug if you find it
does bite.

**How you know you succeeded:** the run prints `=== SMOKE TEST PASSED: SFT+DPO on
OLMoE-1B-7B-0125 ===` as its last line. Short of that, stage 3 succeeding means
`logps/chosen`, `logps/rejected` and `rewards/margin` appear in the step logs
(`dpo_tune_cache.py:599-605`) and a checkpoint lands in `/tmp/olmoe-smoke/dpo`. This is a pipeline
test: no loss value it produces on 32 samples means anything.

### 2. The LoRA-on-MoE `target_modules` policy — Sid asked for this

Both trainers hardcode the same list and neither exposes it as a flag:

```
open_instruct/finetune.py:632
open_instruct/dpo_tune_cache.py:373
    target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
```

On a dense model that is seven modules per layer. On OLMoE it is not, and this is the whole of the
answer to Sid's question. From the HF config (`num_experts: 64`, `num_hidden_layers: 16`,
`hidden_size: 2048`, `intermediate_size: 1024`) and confirmed against the real tensor names in
`model.safetensors.index.json`:

| Matched by | Count | Shape | LoRA params each |
| --- | --- | --- | --- |
| `q/k/v/o_proj`, 4 per layer × 16 layers | 64 | `[2048, 2048]` | `4096·r` |
| `gate/up/down_proj`, 3 per expert × 64 experts × 16 layers | 3,072 | `[1024,2048]`, `[1024,2048]`, `[2048,1024]` | `3072·r` |
| **total adapter pairs** | **3,136** | | **9,699,328·r** |

So r=64, the trainers' default, is **620,756,992** trainable parameters — 8.97% of the 6.92B base,
which is not what anyone means by "parameter-efficient". r=16 is **155,189,248**, or 2.24%. That
is why the run spec sets r=16 and it is the number to quote.

**The router is not adapted, and that is the interesting half.** Each layer's router is
`model.layers.N.mlp.gate` with shape `[64, 2048]`. PEFT matches a list entry by suffix, and
`mlp.gate` does not end in `.gate_proj`, so all 16 routers (2,097,152 params) stay frozen. That is
almost certainly right — moving the router changes which experts fire and would make a LoRA run
non-comparable to the base model's routing — but it is currently true by accident of naming rather
than by decision, and nothing in the repo says so.

Reproduce all of it without downloading weights:

```bash
curl -sL https://huggingface.co/allenai/OLMoE-1B-7B-0125/raw/main/model.safetensors.index.json \
  | python3 -c "
import json,sys,re,collections
c=collections.Counter(re.sub(r'\.\d+\.','.N.',k) for k in json.load(sys.stdin)['weight_map'])
for k,v in sorted(c.items()): print(f'{v:6}  {k}')"
```

**The decision Sid needs** is whether "LoRA on an MoE" should keep meaning "every expert", or
should mean attention-only, or expert-FFN-only, or a shared-expert subset. Attention-only would be
64 adapters and 262,144·r params — 4.2M at r=16, small enough to be genuinely cheap, and it
sidesteps the question of whether adapting 64 experts that each see roughly 1/8 of the tokens
trains them evenly at all. Nobody here has measured that. Making it a flag is a one-line change in
each trainer; deciding what the default should be needs an experiment.

### 3. Should `--load_balancing_loss` be on for MoE post-training?

The run spec leaves it **off**, on the grounds that AI2's own archived OLMoE-0125 recipe never set
it and a smoke test should not confound a pipeline failure with a flag nobody here has run. That
reasoning holds for a smoke test and stops holding immediately after. Before you ablate it, three
things about how it is actually wired, all of which surprised me:

**The two trainers do not do the same thing when you set it.**

- **DPO** adds it explicitly: `dpo_tune_cache.py:570` passes `output_router_logits=args.load_balancing_loss`,
  and `:581-583` do `loss += args.load_balancing_weight * aux_loss`. Default weight **0.001**
  (`dpo_utils.py:109`).
- **SFT** does not add anything itself. `finetune.py:818-820` calls the model with
  `output_router_logits=True` and then `.detach()`s the aux loss into a logging accumulator. The
  aux loss still reaches the gradient — but from *inside* HF's `OlmoeForCausalLM`, which does
  `loss += self.router_aux_loss_coef * aux_loss` (transformers v4.48.1,
  `modeling_olmoe.py:1277`). That coefficient is **0.01**, from the model's own config, not from
  our flag.

**`--load_balancing_weight` is dead code in `finetune.py`.** It is declared at `finetune.py:270`
with a default of 0.5 and never read anywhere else in the file. Grep it. So the SFT weight is 0.01
whatever you pass, the DPO weight is 0.001 unless you pass something, and the SFT flag advertises
0.5. Three different numbers, one of them fictional. Whatever you conclude about the ablation,
this asymmetry is worth a PR on its own.

**How you know you succeeded:** an `aux_loss` key appears in the logged metrics
(`finetune.py:946`, `dpo_tune_cache.py:665`) and the answer is stated as a recommendation with a
number behind it, not as "it seemed fine". The measurable outcome is expert utilisation — whether
the aux loss actually flattens the router's expert distribution at post-training scale, or whether
at 64 experts and a few thousand samples it just adds noise.

---

## Not done, and do not claim otherwise

- **No stage has executed.** No GPU, no checkpoint, no loss curve. Everything above is read out of
  source, config and the HF model index.
- The parameter counts are derived from real tensor shapes but were **not** confirmed against
  `model.print_trainable_parameters()`, which is what PEFT actually prints. That happens on the
  first real run and is the cheapest possible check that the arithmetic is right.
- `--load_balancing_loss` has never been set on this model in this org, by anyone, as far as I can
  find. There is no baseline to compare an ablation against.
- The dataset choices in both stages are smoke-test-sized stand-ins picked for being small and
  public. They are not the OLMoE-0125 preference mixture and no conclusion about DPO quality
  should be drawn from them.
- I have not verified that `merge_lora.py` handles 3,072 expert adapters without exhausting host
  memory. It is stage 2 and therefore mine, but it is directly upstream of you, so if stage 3
  never starts, look there first.

---

## Recommendation

**Get the branch pushed before you touch anything.** Ask Sid — he has push on this fork, he is
already in this thread for the LoRA question, and it costs him thirty seconds. Every item above is
cheap once a run can go and worthless until one can.

**Then run the three stages exactly as committed, once, and change nothing.** The value of the
smoke test is a clean yes/no on whether OLMo's default SFT→DPO path works on an MoE at all. Both
flag questions are more interesting and both should wait until there is a passing baseline to
compare against, or you will not be able to tell a flag effect from a pipeline bug.

**Of the two flag questions, `target_modules` is the one with a deadline** — Sid asked for it and
the answer changes what "LoRA on the MoE" costs by a factor of 37 between attention-only and
everything. `--load_balancing_loss` is a real ablation that needs real compute and should be
scoped as its own piece of work, not smuggled into a smoke test.
