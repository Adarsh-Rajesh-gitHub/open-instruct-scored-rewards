# Slack draft — post-training channel

Paste as-is. Three blockers that affect everyone, not just the smoke test.

---

Update on the OLMoE-1B-7B-0125 SFT+DPO smoke test, plus three things that are going to hit
everyone here and not just me.

The three-stage run (LoRA SFT → merge → LoRA DPO) is written and committed on
`edullm/olmoe-sft-dpo-smoke`. `edullm check` returns `refused: false` with automatic approval on
one L40S, priced at a $5.58 ceiling over 3 hours. Nothing has run yet — I have pull but not push
on both `edu-llm/open-instruct` and `edu-llm/open-instruct-scored-rewards`, and the platform builds
an image only from a commit pushed to an `edullm/**` branch. Filed as `edu-llm/platform#435`. If
anyone with push on either fork can land the branch, the run goes today.

**1. `edu-llm/open-instruct` is not registered with the platform.** `edullm check` there refuses
with `unregistered_repository`. `open-instruct-scored-rewards` is registered, so that is where my
smoke test lives — which means the team's default SFT+DPO is currently sitting in Sophia's GRPO
fork. Registration is already written but blocked behind `edu-llm/open-instruct#4` (Frank's
`.edullm/Dockerfile`), which has to merge first or the platform's daily audit fails with
`registered_dockerfile_is_absent`. That PR is currently red on a uv version pin in
`publish / Verify source identity`.

**2. The 64 H100s aren't reachable on this route.** `gpu-1xh100` and `gpu-8xh100` are both
`provisioned: false` and refuse with `unprovisioned_compute_profile`. Separately, every compute
profile in the catalog is single-node, so there is no way to ask for 64 of anything. The largest
shape that actually places is `gpu-8xa100` — eight A100 80GB on one node, accepted, but
`capacity.yaml` records a 61-minute median queue wait and a 12.6-hour worst case. The route to
anything bigger looks like a purchased capacity block (`edu-llm/platform#413`, unmerged), which
routes to exception approval by an admin. @meric — worth a few minutes on where the 64 H100s
actually live, because it isn't this control plane.

**3. My runs bill to `memory-split`, not post-training.** `edullm check` reports
`"team": "memory-split"` with `"team_source": "from the roster, not from you"`, and
`wandb_project: memory-split` with it. `config/organization.yaml` has me in `memory-split` and not
in `post-training`, and it's the roster that decides, not the submit flag. Same for Stephen. So any
post-training work either of us submits lands on Memory's budget and in Memory's W&B project until
the roster changes. Not urgent for a $5.58 smoke test, flagging it before anything bigger.

Handoff docs for the DPO leg and for the two platform items are on the same branch under
`docs/handoff/`.
