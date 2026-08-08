# Getting `edu-llm/open-instruct` onto the platform, and finding the real route to big GPUs

**From:** Adarsh Rajesh
**To:** the person Slack shows as `ethan.feng` — the roster's only Feng is **Eve Feng
(`NotAnAlgorithm`)**, so this is addressed to them. If that is the wrong person, say so and I will
re-address it; I could not resolve the Slack handle against `config/organization.yaml` and did not
want to guess silently.
**Covers:** Aug 8, 2026
**Read this if:** you are picking up either of the two platform blockers that came out of
smoke-testing SFT+DPO on `allenai/OLMoE-1B-7B-0125`.

---

## Orientation (read first)

Meric asked three of us to smoke-test OLMo's default SFT+DPO on OLMoE-1B-7B-0125 by end of day. I
own the SFT leg and Stephen Zhang has the DPO leg
([`HANDOFF-dpo-and-moe-flags-stephen.md`](HANDOFF-dpo-and-moe-flags-stephen.md)). Building it
surfaced two problems that are not about training at all, are not mine or Stephen's to fix, and
block the whole team rather than one run. Those two are yours.

**Bottom line: the smoke test is written, priced and accepted, and it is running in the wrong
repository on the wrong team's budget.** It has to, because `edu-llm/open-instruct` is not
registered with the platform and `edu-llm/open-instruct-scored-rewards` is. Both of those are
fixable and neither is fixable by me.

### Blocker for you specifically

You will hit the same wall I did the moment you try to push a fix:

```bash
gh api repos/edu-llm/open-instruct --jq .permissions
# {"admin":false,"maintain":false,"pull":true,"push":false,"triage":false}
```

I have pull and not push, and so does Stephen. **No GitHub team grants access to either
open-instruct repo** — `orgs/edu-llm/teams/post-training/repos` and
`.../memory-split/repos` each return only `platform`, and `team-members` returns `OLMo-core` and
`platform`. So access there is a direct collaborator grant. Check your own before you plan around
it; you are on `post-training` and I am not, so yours may differ from mine and I cannot read
anyone's but my own.

Both items below end in a change Frank Gonzalez (`philote-dev`) has to merge, so getting him in the
loop early is the actual critical path, not the engineering.

---

## Item 1: register `edu-llm/open-instruct`

### What is true today

```bash
cd ~/AlphaProject/open-instruct && edullm check --json --experiment olmoe-sft-dpo-smoke --dataset none
```

```json
"refusals": [{"code": "unregistered_repository",
  "detail": "run edullm add repository --reason '<why>' to register 'open-instruct' ...
    Registered today: OLMo-core, edullm-alt-cl, edullm-data, edullm-p1, olmo-eval-full,
    open-instruct-scored-rewards."}],
"refused": true
```

`config/repositories.yaml` on `edu-llm/platform` main carries six entries and `open-instruct` is
not one of them. `open-instruct-scored-rewards` is, which is the only reason my smoke test has a
home at all — and it is Sophia's GRPO fork, which is a strange place to keep the team's default
SFT+DPO runs.

**This matters beyond my run.** Frank has said post-training moves to `edu-llm/open-instruct`.
Every person who tries to submit from there hits `unregistered_repository`, and the refusal is
correct, so there is nothing to work around.

### This is already in flight and it is stuck, which changes the job

Do not start by opening a registration PR. One is already written and blocked behind a
prerequisite. From Frank's commit message on
[`edu-llm/open-instruct#4`](https://github.com/edu-llm/open-instruct/pull/4), opened 2026-08-06:

> the eduLLM platform cannot build this repository until `.edullm/Dockerfile` exists on main. Its
> registration is written and cannot merge before this does: the platform's daily audit has a
> registered-dockerfiles job that fails with `registered_dockerfile_is_absent` when a registration
> names a file the repository does not have, so the image definition has to land first.

That is the ordering constraint: **image definition first, registration second.** It matches the
CLI, where `edullm add repository` takes `--dockerfile`, "repository-relative path to the
Dockerfile the build workflow builds".

PR #4 is `MERGEABLE` and `OPEN` with no review, and one check red:

```bash
gh pr view 4 --repo edu-llm/open-instruct --json state,mergeable,statusCheckRollup
# publish / Verify source identity  -> FAILURE
```

The failure is not in the Dockerfile. The workflow step pins a uv version and the runner disagrees:

```bash
gh run view 31059896624 --repo edu-llm/open-instruct --log-failed | grep -i uv_version
# echo "unexpected_uv_version" >&2
# echo "Expected uv ${UV_VERSION} but PATH answers with: ${installed}" >&2
```

`.edullm/Dockerfile` exists on the branch (11,623 bytes) and is absent from main:

```bash
gh api repos/edu-llm/open-instruct/contents/.edullm/Dockerfile                              # 404
gh api 'repos/edu-llm/open-instruct/contents/.edullm/Dockerfile?ref=edullm/add-research-image'  # 200
```

### What you own

Getting PR #4 green and merged, then getting the registration PR opened and merged behind it. The
uv pin is a small, self-contained CI fix and is almost certainly the whole of what stands in the
way — but it is Frank's PR, so the move is to tell him what the failing step says rather than to
push at it. If he wants it taken off his hands, the fix belongs on his branch and needs push there.

Once `.edullm/Dockerfile` is on main:

```bash
cd ~/AlphaProject/open-instruct
edullm add repository --reason 'post-training runs the default SFT+DPO path from this repository' \
                      --dockerfile .edullm/Dockerfile --json
```

That opens a pull request against `edu-llm/platform`; it does not grant anything, and it needs a
reviewer. Do not pass `--force` to anything, and do not edit `config/repositories.yaml` by hand.

**How you know you succeeded:**

```bash
cd ~/AlphaProject/open-instruct && edullm check --json --experiment <slug> --dataset none
```

exits 0 with no `unregistered_repository` in `refusals`, and `edu-llm/open-instruct` appears in the
"Registered today" list that the refusal itself prints. Note that `edullm check` writes a first
`.edullm/run.yaml` when there is none and says so **on stderr** — read stdout on its own, or that
note becomes a JSON parse error.

---

## Item 2: the 64 H100s Meric referenced

### They are not reachable on this route, and the gap is bigger than "out of stock"

Both H100 profiles refuse, for free, before anything is dispatched:

```bash
edullm check --json --experiment olmoe-sft-dpo-smoke --dataset none --compute gpu-8xh100 --hours 3
```

```json
{"code": "unprovisioned_compute_profile",
 "detail": "compute profile 'gpu-8xh100' is priced in the catalog but no compute environment is
   provisioned for p5.48xlarge. Provisioned today: cpu-32vcpu, gpu-1xa10g, gpu-1xl4, gpu-1xl40s,
   gpu-1xt4, gpu-4xa10g, gpu-4xl4, gpu-4xl40s, gpu-4xt4, gpu-8xa100, gpu-8xa10g, gpu-8xl4,
   gpu-8xl40s, gpu-8xt4."}
```

`gpu-1xh100` (p5.4xlarge, $6.88/hr) gives the identical refusal. Both are `provisioned: false` in
`config/workload-catalog.yaml`: priced, selectable, and with no queue behind them.

**There is also no such thing as 64 cards here.** All 17 compute profiles in the catalog are
`nodes: 1`, and the largest instance type in any of them holds 8 accelerators. "64 H100s" is 8
nodes of `gpu-8xh100`, and this platform has no multi-node profile of any shape, provisioned or
not. So the ask is two independent gaps: the shape is unprovisioned, *and* the platform cannot
express eight of it. Worth being precise about that with Meric — if the 64 H100s are real, they are
real somewhere that is not this control plane, and finding out where is most of this task.

### The largest thing that actually places today

Measured, not assumed — this is a join of `config/capacity.yaml` against
`config/workload-catalog.yaml` in the installed `edullm 4.5.0` config:

| Profile | Instance | Provisioned | Places | $/hr |
| --- | --- | --- | --- | --- |
| `gpu-1xl40s` | g6e.xlarge | yes | after a wait | 1.86 |
| `gpu-4xa10g` | g5.12xlarge | yes | after a wait | 5.67 |
| `gpu-1xh100` | p5.4xlarge | **no** | unreliably | 6.88 |
| `gpu-4xl40s` | g6e.12xlarge | yes | after a wait | 10.49 |
| `gpu-8xa10g` | g5.48xlarge | yes | after a wait | 16.29 |
| **`gpu-8xa100`** | **p4d.24xlarge** | **yes** | **after a wait** | **21.96** |
| `gpu-8xl40s` | g6e.48xlarge | yes | unreliably | 30.13 |
| `gpu-8xh100` | p5.48xlarge | **no** | unreliably | 55.04 |

**`gpu-8xa100` is the ceiling: eight A100 80GB on one node, and it is accepted.** I checked, and it
comes back `refused: false` with approval class `automatic` at $65.87 for three hours. Its
`capacity.yaml` entry is the thing to read before promising anyone a runtime:

> Fourteen nodes arrived between 2026-08-03 and 2026-08-05, and the thirteen runs submitted on this
> shape waited a median of 61 minutes from submission to start, with a worst observed case of 12.6
> hours. None was ever cancelled for want of capacity.

An hour's median wait is survivable; 12.6 hours is not, if someone is holding a deadline against it.

### The real route, which is capacity blocks

This is the part worth your time, and I would not have found it from the CLI. Two open PRs on
`edu-llm/platform` build the path from a purchased AWS capacity block to a runnable queue:

- [`#413`](https://github.com/edu-llm/platform/pull/413) "Make a purchased capacity block reach an
  admin, a queue and a targeted launch" — adds four block-backed shapes: `gpu-8xa100-80gb`,
  `gpu-8xh200`, `gpu-8xb200`, `gpu-8xb300`. All carry `capacity_block_backed: true`, which routes a
  submission to `ApprovalClass.EXCEPTION` and therefore to a platform admin. All stay
  `provisioned: false` until a block is actually bought.
- [`#416`](https://github.com/edu-llm/platform/pull/416) "Promote gpu-8xb200 before its block is
  bought" — stacked on #413, and its description says a `p6-b200.48xlarge` block "may be bought
  this weekend".

Neither is merged, and none of those four shapes exists in the config `edullm 4.5.0` installs —
`rg 'b200|h200|b300|capacity_block_backed' <config-dir>/workload-catalog.yaml` returns nothing. So
today they are not selectable at all.

The shape of the answer for Meric is therefore: **the route to eight big cards is a purchased
capacity block plus an exception approval from a platform admin, not a profile anyone can select**;
the nearest thing to H100s on that path is `gpu-8xh200`, not H100 at all; and even that is one node
of eight. Frank owns both PRs and is the person who knows whether a block is being bought.

### What you own

Establishing what Meric actually needs 64 H100s *for*, and which of these is the honest answer:
`gpu-8xa100` today with an hour's queue, a capacity block behind #413 with an admin's approval, or
compute outside this platform entirely. Get the number and the deadline from Meric before pricing
anything.

**How you know you succeeded:** a named shape, a named route, and a price that came out of
`edullm check --json` rather than out of a document. Do not quote a price, a runtime bound or an
approver from this file — read `cost` and `approval_class` out of `check`, because the reviewed
configuration behind them changes without anyone being told.

---

## Not done, and do not claim otherwise

- I have not confirmed who holds push on either open-instruct repo. The API refuses the question
  without push access (`403 Must have push access to view repository collaborators`). What I can
  say is that `sidvenkatayogi`, `zsophiaaa`, `pianomaster99` and `philote-dev` have each pushed an
  `edullm/**` branch to `open-instruct-scored-rewards`, so they demonstrably have it there.
- I have not verified that the uv-version failure is the *only* thing wrong with PR #4. It is the
  only red check, but "Run unit tests" and "Run GPU tests" both show `CANCELLED`, so they never
  reported.
- I have not asked Frank about any of this. Nothing below has been raised with him yet.
- Nothing in either item was tested by dispatching anything. Every `edullm check` above is free,
  reaches no network and dispatches nothing. I did not run `submit`, `run` or `shell`.

---

## Recommendation

**Take registration first.** It is the shorter task, it is already 90% written by Frank, and it
unblocks every person on the team rather than one run — including mine and Stephen's, which are
currently squatting in Sophia's GRPO fork because it is the only registered repository that
contains the trainers we need.

**Treat the H100 question as a scoping conversation, not an infrastructure task.** The verified
facts are that both H100 profiles are unprovisioned, that no multi-node profile exists, that the
largest thing that places is one node of eight A100s, and that the only path to anything bigger is
a purchased capacity block gated behind an admin. That is enough to go back to Meric with a real
answer, and doing so is worth more than any amount of further digging in the config.

**Do not write anything that calls AWS.** No `boto3`, no `aws` CLI, no `curl` at an AWS endpoint.
The credentials live in workflows pinned to `main` and a laptop cannot get one; for the few people
where it would succeed, it produces a run nobody can cite.
