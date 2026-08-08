# Patches to upstream open-instruct

Everything added by this fork lives in three new directories:

- `open_instruct/scored_rewards/` — the generic score-based reward layer
- `open_instruct/spec_decode/` — speculative decoding for the rollout engine
- `projects/` — one project's specifics, imported by nothing in `open_instruct/`

Upstream files are touched in **six places**, kept deliberately small so
rebasing onto `allenai/open-instruct` stays a non-event. With none of the new
flags set, behaviour is unchanged — and that claim is enforced by a test, not
just asserted (`test_integration.py::test_no_flags_returns_the_plain_upstream_config`).

```
README.md                             | 20 +++++++++++++++++++   docs only
open_instruct/data_loader.py          | 18 ++++++++++++++++++    six new flags
open_instruct/grpo_fast.py            | 23 ++++++++++-------------  two call sites
open_instruct/data_loader.py          | 25 +++++++++++++++++++    five specdec flags
open_instruct/vllm_utils.py           | 39 ++++++++++++++++++--   engine + drain
open_instruct/grpo_fast.py            |  8 ++++++++             call site + metrics
open_instruct/benchmark_generators.py |  2 ++                   one kwarg
conftest.py                           |  3 +++                  two test skips
```

Every added file is under `open_instruct/scored_rewards/`,
`open_instruct/spec_decode/` or `projects/`, all new directories, so
`git diff --stat` against upstream separates the fork's code from upstream's
without reading any of it.

---

## 1. `open_instruct/data_loader.py` — six CLI flags

Six optional fields on `StreamingDataLoaderConfig`, all defaulting to off:
`reward_plugins`, `group_scorer`, `group_reward_mode`, `group_reward_scale`,
`group_scorer_strict`, `score_verifiers`.

They go on `StreamingDataLoaderConfig` rather than `Args` because that is the
config already threaded into reward construction, and because `ArgumentParserPlus`
derives the CLI from these dataclasses — new fields become new flags with no
parser changes.

**Conflict risk on rebase: nil.** Added fields, changed none.

## 0. `README.md` — one section

A "Score-based rewards (fork addition)" section under the RLVR heading, marked
as not-upstream and pointing at `PATCHES.md`. Documentation only. It is here so
that someone who clones the fork and reads the front page discovers the addition
instead of finding two unexplained directories.

**Conflict risk on rebase: low**, and a README conflict is never subtle.

## 2. `open_instruct/grpo_fast.py` — one call swapped, one call added

**(a)** The literal `RewardConfig(...)` construction in `main` becomes
`make_reward_config(args, streaming_config, tools_config)`. That function
contains the identical constructor call and returns it unchanged unless
`--group_scorer` or `--score_verifiers` is set, in which case it returns a
`GroupRewardConfig` subclass.

A subclass rather than an edit to `ground_truth_utils.RewardConfig` because
`RewardConfig` is shared by every trainer in the repo, and this behaviour is
GRPO-specific: `GroupScorer` needs all G samples of a prompt, which only the
grouped rollout path has.

**Conflict risk on rebase: low but real.** If upstream adds a field to
`RewardConfig`, the constructor inside `make_reward_config` needs the same field.
The symptom is an obvious `TypeError` at startup, not silent drift.

**(b)** `load_reward_plugins(streaming_config.reward_plugins)` as the first
statement in `main`.

It has to be first. Plugins register environments as well as scorers, and
`initialize_tools_and_envs` reads `TOOL_REGISTRY` further down — a plugin
imported any later would register into a registry that had already been read.

**Conflict risk on rebase: nil.** One line at the top of a function.

---

## What was deliberately *not* patched

**`vllm_utils.compute_rewards` was left alone.** It already loops per prompt with
that prompt's whole group in `result.responses`, so the group is reachable from
inside `RewardConfig` without touching the caller. The group scorer runs once per
prompt from `GroupRewardConfig` and caches its result for the individual sample
calls that follow.

**No LoRA.** `use_peft` is declared in `model_utils.py` and referenced nowhere in
`grpo_fast.py`; wiring PEFT through DeepSpeed and the vLLM weight sync is a real
change to the training loop, not a patch, and it is out of scope here. The
consequence is that a 3B policy will not fit on one 80GB card beside vLLM, so
single-GPU users need to serve the environment and judge models externally.

**No changes to the advantage computation.** `normalize_then_sum` in the reward
produces zero-mean-within-group scores, so upstream's default
`advantage_normalization_type=centered` is already the right arithmetic.

---

# Speculative decoding for the rollout (`open_instruct/spec_decode/`)

Turns on vLLM's EAGLE-3 speculative decoding inside the RL rollout engine, to
measure the acceleration reported in arXiv:2604.26779 on an OLMoE policy. It is a
*lossless* accelerator: rejection sampling makes the accepted tokens the target
policy's own samples, so this changes rollout throughput and nothing about what is
sampled or optimised.

New files, none imported unless a flag is set:

| file | what |
|---|---|
| `spec_decode/config.py` | four CLI scalars → vLLM's `speculative_config` dict. No vLLM import |
| `spec_decode/olmoe_eagle3.py` | OLMoE target that emits EAGLE-3 auxiliary hidden states |
| `spec_decode/registration.py` | lazy `ModelRegistry` override. Imports only `ModelRegistry` |
| `spec_decode/metrics.py` | vLLM stat logger that accumulates acceptance counters |
| `spec_decode/reporting.py` | per-step R_gen, α, and the paper's §2.2 speedup bound |

## 4. `open_instruct/data_loader.py` — five more CLI flags

Five optional fields on `VLLMConfig`, all defaulting to off:
`vllm_speculative_method`, `vllm_speculative_model`,
`vllm_num_speculative_tokens`, `vllm_speculative_draft_tensor_parallel_size`,
`vllm_collect_spec_decode_stats`; plus one method, `speculative_config()`, which
returns `None` unless the first is set.

On `VLLMConfig` rather than `Args` for the same reason the reward flags are on
`StreamingDataLoaderConfig`: it is the config already threaded to
`create_vllm_engines`, and `ArgumentParserPlus` derives the CLI from these
dataclasses, so new fields become new flags with no parser changes.

**Conflict risk on rebase: nil.** Added fields, changed none.

## 5. `open_instruct/vllm_utils.py` — one param, one kwarg, two hooks

**(a)** `create_vllm_engines` takes `speculative_config` and
`collect_spec_decode_stats`, both forwarded to the actor. `speculative_config`
reaches `AsyncEngineArgs` through the existing `**kwargs`; `None` is that class's
own default, so an unflagged run builds the same engine as before.

**(b)** `_setup_and_start_async_engine` registers the EAGLE-3 target before the
engine exists, when and only when a speculative config was passed. It has to be
before, and in this process: the engine core runs here
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) and its workers are forked from it, so a
lazy registration made here is the one they resolve through.

**(c)** `from_engine_args` gains `stat_loggers`, and the actor gains
`drain_spec_decode_metrics()`. A custom stat logger re-enables stats for itself
despite `disable_log_stats = True` — `AsyncLLM` does
`log_stats or has_custom_loggers` — without restoring the periodic throughput
logging that flag was set to silence.

**Conflict risk on rebase: low.** If upstream changes the `.remote(...)` kwarg
list or moves the `AsyncEngineArgs` construction, the symptom is a `TypeError` at
engine startup, not silent drift.

## 6. `open_instruct/grpo_fast.py` — one param, one dict entry

`one_training_step` takes `vllm_engines` (keyword, defaults to `None`) and merges
`spec_decode.reporting.step_metrics(...)` into the metrics dict. Returns `{}`
unless `--vllm_collect_spec_decode_stats`, so an unflagged run logs exactly what
upstream logs.

**Conflict risk on rebase: low.**

## 7. `open_instruct/benchmark_generators.py` — one kwarg

Passes `speculative_config` through, because the generation-only sweep that
decides whether any of this is worth running at RL scale is built on this script.

**Conflict risk on rebase: nil.**

## What was deliberately *not* patched

**vLLM is not forked or rebuilt.** `OlmoeForCausalLM` gains EAGLE-3 support by
`ModelRegistry.register_model` pointing the architecture key at a subclass, so
the pinned `vllm==0.21.0` wheel is untouched. The registration is a
`"<module>:<class>"` string rather than a class object, which is vLLM's documented
way to avoid initialising CUDA in a parent that forks.

The entry-point plugin mechanism (`vllm.general_plugins`) would be the tidier
route and does not work here: the platform image does `COPY . /opt/open-instruct`
and sets `PYTHONPATH`, never `pip install`, so there is no dist-info metadata for
`importlib.metadata.entry_points()` to find. That failure would be silent.

**The draft model is not trained during RL, and the learner is untouched.** vLLM
keeps target and draft weight transfer on separate calls —
`start_weight_update()` and `start_draft_weight_update()` — and open-instruct only
calls the first, so the draft stays frozen with no code required. That is the
paper's "offline" mode, worth 1.77× → 1.78× against an in-domain draft (its
Table 5). Online adaptation would need hidden-state capture and a detached draft
loss inside the DeepSpeed learner, and is out of scope.

**`--use_vllm_logprobs` must stay off** for any run whose loss matters. It is not
patched and does not need to be, but with it on the loss would consume logprobs
returned by an engine that is speculating; leaving it off forces learner-side
recomputation and makes verifier-exactness structural rather than an assumption.

---

## Rebasing

```bash
git remote add upstream https://github.com/allenai/open-instruct.git
git fetch upstream && git rebase upstream/main
python -m unittest \
    open_instruct.scored_rewards.test_scored_rewards \
    open_instruct.scored_rewards.test_integration
```

No GPU, no ray, no vLLM, no network. The second one is the one that matters
after a rebase: it builds a real `RewardConfig`, calls `.build()`, and invokes
the result with the exact argument list `vllm_utils.compute_rewards` passes, so
a changed reward signature or a new `RewardConfig` field fails there rather than
twenty minutes into a GPU run. It needs open-instruct's own dependencies
installed and skips cleanly when they are not, which is why the other two files
avoid them entirely.
