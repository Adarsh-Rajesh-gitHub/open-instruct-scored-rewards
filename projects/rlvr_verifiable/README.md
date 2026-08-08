# RLVR on maths, code and factual QA

GRPO against rewards that can be checked rather than learned. Maths answers verified against a
reference, code verified by running it, short-answer QA verified by overlap with the reference.

**Almost nothing here is new code.** open-instruct already ships every verifier this needs, and
`projects/pedagogy_rm` already has the cluster plumbing. This directory is a dataset choice, four
launchers, and two scripts that check things before a GPU is spent.

## What is reused, and from where

| Piece | Where it comes from | Written here? |
|---|---|---|
| Training loop | `open_instruct/grpo_fast.py` | no |
| Maths verifiers | `math`, `gsm8k`, `strict_math` in `ground_truth_utils.py` | no |
| Code verifiers | `code`, `code_stdio`, same file | no |
| Factual verifiers | `string_f1`, `string_matcher`, same file | no |
| Code execution service | `open_instruct/code_utils/api.py` | no |
| Datasets | Ai2's published `allenai/RLVR-*` mixes | no |
| LoRA / single-GPU path | `grpo_fast.py` + `--single_gpu_mode` | no |
| Slurm wrapper | `projects/pedagogy_rm/scripts/train.sbatch` via `INNER` | no |
| The `grpo_fast.py` invocation | `scripts/train.sh` | **yes** |
| Per-domain launchers | `scripts/train_{math,code,factual}.sbatch` | **yes** |
| Dataset prefetch + verifier audit | `prefetch.py` | **yes** |

There is **no reward model, no plugin and no scorer**, which is the whole difference from
`projects/pedagogy_rm`. That project needed a fitted probe because "was that good teaching" has no
verifier. Here the answer is checkable, so the reward is `--apply_verifiable_reward True` and data
with the right `dataset` field.

## How a row selects its verifier

Every row is three fields, and the `dataset` field is the verifier's registered name:

```json
{"messages": [{"role": "user", "content": "..."}],
 "ground_truth": "42",
 "dataset": "gsm8k"}
```

`apply_verifiable_reward` looks that name up and calls it. A row may carry lists in both fields to
run several verifiers and sum their weighted scores.

The 25 registered names, for the three domains here:

| Domain | Names | Checks |
|---|---|---|
| maths | `math` | final answer vs reference, latex and sympy aware |
| | `gsm8k` | the integer after `####` |
| | `strict_math` | as `math`, without the lenient fallbacks |
| code | `code` | runs extracted python against assert-style tests |
| | `code_stdio` | runs it against (stdin, expected stdout) pairs |
| factual | `string_f1` | token F1 against the reference |
| | `string_matcher` | exact match after normalisation |

**A name that is not registered is skipped with only a log warning.** Those rows score zero and drag
the mean, so a mix that is 30% unregistered looks like a run that cannot learn. `prefetch.py --report`
exists to catch that before training.

## Running it

```bash
# 1. On the login node, where there is a network. The Slurm wrapper pins HF_HUB_OFFLINE=1, so
#    anything not cached fails at startup instead of downloading a surprise revision.
python projects/rlvr_verifiable/prefetch.py --mix math --report

# 2. Train. Each launcher points the pedagogy Slurm wrapper at this project's train.sh via INNER.
sbatch projects/rlvr_verifiable/scripts/train_math.sbatch
sbatch projects/rlvr_verifiable/scripts/train_code.sbatch
```

Everything is env-overridable, so a smoke run needs no new file:

```bash
EXP=rlvr_smoke EPISODES=1600 PROMPTS=8 SAMPLES=4 SAVE_FREQ=0 \
  sbatch projects/rlvr_verifiable/scripts/train_math.sbatch
```

Defaults: OLMo-2-7B-Instruct, LoRA r=32, lr 4e-5, 32 prompts x 8 samples, 2048 in / 2048 out packed
to 4096, one H100 with the learner and vLLM colocated.

## Domain notes worth reading before trusting a number

**Maths is the clean case.** The verifier compares against a reference answer and is hard to satisfy
by accident. If reward rises here it is because more answers are right. This is the domain to debug
the pipeline on.

**Code needs a service, and a silent failure mode.** `CodeVerifier` POSTs to an HTTP endpoint; if it
is not up, every rollout scores zero and the run reads as "cannot learn" rather than "cannot connect".
`train_code.sbatch` therefore starts the bundled executor on `127.0.0.1`, health-checks it, and then
runs a passing and a failing program through it and asserts the results differ — verified working
locally before this was committed. It also traps EXIT so a preemption does not leave an orphan holding
the port.

It is still executing model-written code. `api.py` uses a subprocess with a 1-second timeout, which
bounds runtime but is not a sandbox — no filesystem or network isolation. For unattended runs prefer
`open_instruct/code/Dockerfile`.

**Factual has the weakest verifier and is the one to distrust.** Token F1 rewards emitting more of the
right words, which padding an answer can do without knowing more. `string_matcher` is stricter but
scores "Paris" against "Paris, France" as 0, which is too sparse to learn from. If reward climbs while
answers get longer, suspect the metric before believing the result — this is the same failure the
pedagogy project spent weeks on, where a reward correlated −0.60 with length and most of the measured
gain turned out to be brevity.

`train_factual.sbatch` deliberately has no default dataset. Run
`prefetch.py --mix general --report` and read which verifiers the rows actually name before choosing
one.

## What has not been done

- **No run yet.** These launchers are untested on the cluster; only the code executor and the verifier
  registry lookup have been verified, both locally.
- **No held-out benchmark.** Reward on an eval split is not the same as accuracy on GSM8K's or
  MATH's own test set. `projects/pedagogy_rm/benchmark.py` already scores GSM8K and MMLU behind a
  `--adapter` flag and is the obvious thing to point at a checkpoint from here.
- **No baseline.** The pedagogy project learned this the hard way: the untrained model and a
  *prompted* untrained model are different controls, and only the second one is convincing. For maths
  that means comparing against base with a chain-of-thought prompt, not bare base.
