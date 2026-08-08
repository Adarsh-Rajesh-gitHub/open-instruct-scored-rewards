# What is wrong with this setup

A deliberately adversarial read of the pedagogy reward-model project, written against the state of
things on 2026-08-06. Ordered by how much each item threatens the conclusions, not by category. The
first four would change what the paper is allowed to claim; the rest weaken evidence or limit scope.

Every number here is recomputed from `data/allhist.json` and `data/head5.npz` rather than quoted from
the report, because two of the report's numbers turned out to be wrong when checked that way.

---

## Tier 1 — these threaten the headline claim

### 1. The probe dimensions left 60% of their available reward on the table

An earlier draft of this document said "most of the reward gain is a hand-written word-count rule,
not the learned probe", on the grounds that the length term supplied ~70% of the improvement. That
number is right but the argument built on it was wrong in two ways, and the corrected version says
something more interesting.

**First, the length term is a legitimate part of the objective, not a cheat.** Length fit is a
human-rated quality with 1131 judgements behind it, and the reward was *designed* to give it a large
share: `2.0 x length_fit` against a 5-dimension mean that maxes at 2.2, so the length term owns
2.0 of a 4.2 maximum — **48% of the objective by construction**. Observing that it ends up at 52% of
the realised reward is close to the design intent, not a discovery.

**Second, comparing raw gains ignores how much room each part had.** The four positive dimensions
start near 2.3–2.7 out of 3 and `leak` starts at 1.77 where lower is better, so the paper is right
that there is not much room. The question is whether that explains the gap. Computed for arm E:

| Component | Start | End | Gained | Best possible | Headroom | Captured |
|---|---|---|---|---|---|---|
| `elicits` | 2.307 | 2.698 | +0.391 | 3.0 | 0.693 | 56% |
| `leak` | −1.770 | −1.450 | +0.320 | −1.0 | 0.770 | 42% |
| `actionable` | 2.338 | 2.609 | +0.271 | 3.0 | 0.662 | 41% |
| `correct` | 2.560 | 2.659 | +0.099 | 3.0 | 0.440 | 22% |
| `targeted` | 2.672 | 2.742 | +0.069 | 3.0 | 0.328 | 21% |
| **5-dim mean** | 1.622 | 1.852 | **+0.230** | 2.2 | **0.578** | **40%** |
| **length x2** | 1.396 | 1.981 | **+0.585** | 2.0 | **0.604** | **97%** |

**The two halves had essentially the same headroom — 0.578 against 0.604.** So "less room to improve"
does not explain the difference. What differs is the fraction taken: the length term captured 97% of
what was available to it, the probe dimensions 40%.

That reframes the flaw, and makes it a sharper one. The paper's central claim is that the ceiling
belongs to the reward model — "its ceiling is what the probe can express, and all three optimisers
reach it." **The headroom numbers say otherwise.** The probe was willing to pay another 0.578 and the
policy collected 0.230 of it. Whatever stopped the runs at 3.84, it was not the probe running out of
expressible range. The candidates are that the probe's signal is uninformative in that region, or
that the behaviour it would pay for is genuinely hard to produce — and those have different
consequences, so the difference matters.

The per-dimension rows also qualify the paper's explanation for `targeted` being flat. It is
attributed to starting at 2.58 with no room; in fact it had 0.328 of room and took 21% of it, the
worst capture rate of the five alongside `correct`.

**Fix:** report the headroom table rather than the raw decomposition, since the raw split invites the
wrong conclusion. Then run the reward ablation that has never been run — probe only, length only,
both — which is the only way to separate "the probe's signal is weak here" from "this behaviour is
hard". And soften the ceiling claim: it is a ceiling on what the *policy reached*, not on what the
reward model could express.

### 2. ~~There is no baseline that could falsify the result~~ — RUN, AND THE RESULT SURVIVED

**Resolved 2026-08-06.** Base was sampled with `Keep your reply between 18 and 40 words.` appended, on
held-out questions. The prompt worked on length (median 35 → 28 words, matching arm E's 27.2) and the
probe score fell slightly, from 1.547 to 1.475. Arm E scores **1.879** at the same length.

So at matched length the trained policy is +0.40 ahead of a prompted base, and the rival explanation
this item was raised to test is dead. The remainder of the item is kept below for the reasoning, which
still applies to any future claim that has no control.



The only comparison anywhere is against the raw untrained model. Since the length term captured 97%
of its headroom against the probe's 40% (item 1), the obvious rival explanation is that
**prompting the base model to write 18–40 words reproduces much of the gain at zero training cost**.
That control does not exist. `sample_ladder.sbatch` draws base with no adapter and no modified prompt;
nothing in the repo tests a length-instructed base.

This is the single cheapest experiment that could overturn the paper, which is exactly why its
absence matters. It is a sampling job, not a training run — a few GPU-hours.

**Fix:** sample base with `Keep your reply between 18 and 40 words` appended, score it with the same
head, and put it in the blind pool. If it lands near arm E, the RL contributed much less than claimed.

### 3. The rubric that training optimised is the one shown to be broken

The paper's ablation section diagnoses the V1 rubric convincingly: `actionable` and `elicits`
correlate at 0.95, the five qualities have an effective rank of 2.9, the first principal component
explains 47% of variance, and the nine turns that max four qualities at once have a median length of
**12 words**. The conclusion drawn is correct — every V1 quality measures what the tutor *withheld*,
so the rubric is maximised by withholding everything.

Then a V2 rubric is built (`guidance`, `locates`, `hands_over`, `verdict`, `leak`, `correct`), and its
inter-rater agreement is validated: all six clear 0.4 where two of the original six did not.

**But no head was ever fitted on V2 and no training run ever used it.** `data/head5.npz` scores
`actionable, correct, elicits, leak, targeted` — V1. Arms E, F and G all optimise the rubric that was
just shown to have a degenerate optimum at 12 words.

What was actually done instead was to bolt a length term on top to suppress the symptom. This connects
to item 1: a rubric that still rewards maximal withholding is one the length band has to actively
fight, which is a plausible reason the probe dimensions only collected 40% of what they offered while
the length term collected 97%. The two flaws may be one flaw, and the reward ablation in item 1 is
what would show it.

The paper presents the rubric rebuild in the results section, which invites the reader to believe it
is part of the pipeline that produced the training numbers. It is not.

**Fix:** either say plainly that V2 is unused and the length term is a symptom patch, or fit a V2 head
and rerun one arm. The second is a few hours of labelling plus one training run and would make the
rubric work load-bearing instead of decorative.

### 4. The validation is largely internal to the rubric

The headline validation is that the probe claimed +0.36 and two panels paid +0.34 and +0.35. But both
panels were asked to score **the same six dimensions the reward optimises**. If the rubric is a poor
proxy for teaching — which item 3 argues it is — then agreement between the probe and raters using
that rubric measures whether the probe reads the rubric correctly, not whether the policy teaches
better. The agreement is real but its object is narrower than the claim.

The only genuinely rubric-independent measurement in the project is the pairwise preference test,
where a person is shown two turns and asked which is better with no dimensions involved. That is 28
decided pairs — underpowered by the paper's own admission, which notes a 64% effect needs about 60.

**Fix:** the remaining pairwise pairs already exist. Finishing them is the highest-value hour in the
project, because it is the only evidence that does not inherit the rubric's assumptions.

---

## Tier 2 — these weaken the evidence that exists

### 5. The evaluation questions were in the reward model's fitting set

45 of the 50 held-out questions were among those the probe was fitted on. The paper says this and
correctly notes the probe is frozen so nothing leaks. But it means every "held-out reward tracks
training reward" statement is measured on questions the reward model has seen, which flatters the
generalisation claim specifically.

### 6. One seed per arm, so the learning-rate ladder cannot support its conclusion

`SEED=1` throughout; no arm was repeated. The ladder's finding is that 2e-5, 4e-5 and 8e-5 reach
3.84, 3.88 and 3.86 — differences of 0.02–0.04. **With one run each there is no way to know whether
that is convergence to a shared ceiling or three draws from the same noise.** The stronger claim in
the paper ("the optimiser chooses how fast, the reward model chooses how far") is plausible and I
believe it, but the experiment as run cannot distinguish it from seed variance.

The same applies to the corrected drift claim (F and G both at 0.29): one seed each.

**Fix:** three seeds at one learning rate would establish the noise floor and cost less than the
ladder already did.

### 7. The strongest-looking result is agent-rated only

The ladder blind evaluation — "every arm beats base on every quality", totals 1.73 / 1.74 / 1.71 — is
five model raters and **no human at all**. The paper flags this, and its own warning is severe: the
same raters got arm C backwards, ranking it below arm A when the human preferred it 29–11, because
they tie answer-leakage to length at +0.59 against the human's +0.43. Arm E is shorter than arm C, so
the bias now runs in arm E's favour.

So the table most likely to be quoted is produced by an instrument that is known to be biased in the
direction of the result.

### 8. The human sample is one person, 17 questions, 28 pairs

Every claim about which policy is better ultimately rests on this. The model panel agrees with her,
which is why either is believable, but the panel's per-turn agreement with her is 0.03 on
`actionable` — so "the panel agrees" is not independent corroboration of her ranking, it is a
different instrument that happens to land in the same place on aggregates.

There is also no inter-rater reliability for the human against herself: no turns were re-presented to
measure her own consistency, so we cannot separate her signal from her noise.

---

## Tier 3 — structural choices that limit what the result can mean

### 9. The policy never interacts with a student

Each training prompt is a frozen `(question, dialogue-so-far)` snapshot. The policy writes one turn
and is scored on it. Two consequences:

- **No credit assignment over a conversation.** A turn that sets up a good next exchange scores
  identically to one that does not.
- **The contexts are off-policy and stay off-policy.** Every `dialogue-so-far` was generated by the
  *base* model tutoring a Qwen-1.5B student. As the policy drifts, it is being trained to respond to
  conversation histories that its own behaviour would never produce. At 0.25 nats and 46→28 words the
  drift is small, so this is currently mild — but it grows with any longer or harder training, and
  it is the mechanism that will bite first if someone extends this.

### 10. The encoder and the policy are the same model

The probe reads activations from a frozen OLMo-2-7B-Instruct; the policy is that same checkpoint plus
an adapter. Freezing the encoder correctly closes the representation-hacking channel. But it leaves a
subtler issue: the probe was fitted on activations of *base-model* text and is applied to
increasingly drifted text.

This is not hypothetical — it is the documented cause of the one failure the paper reports honestly.
The `leak` head ties leakage to length at +0.70 out of distribution against 0.44 in distribution, and
two principled repairs both failed because they were estimated in distribution. The paper treats this
as a `leak`-specific problem. It is a property of the whole design and should be expected on every
dimension.

### 11. The reward averages dimensions that are not independent

Averaging five V1 qualities with an effective rank of 2.9 means the reward counts roughly three
things while appearing to count five, weighting them by an accident of how correlated they are. Arm B
was the attempt to address this by z-scoring, and it failed for an unrelated reason (it raised the
effective step size 3.6x as a side effect), so the question is still open and is now buried.

### 12. `correct` is in the reward at r=0.63

Factual correctness is a verification problem. A linear probe reading it at 0.63 from activations is
weak, and rewarding a weak correctness signal plausibly teaches *confident-sounding* rather than
*correct*. Nothing in the evaluation would detect that: the human rates `correct` against a generated
worked solution, not an authoritative one.

### 13. The simulated student may not resemble a student

The dialogues were generated with Qwen-2.5-1.5B-Instruct prompted to make plausible mistakes. The
project's own earlier finding is that simulated students are a trap — the whole reason the reward
reads the tutor's turn is that student outcome carried no pedagogical signal. That argument retired
the student from the *reward*, but the student still generates every context the tutor is trained on
and every context it is evaluated on. Its failure modes shape what "targeted" and "locates" even mean
here, and none of that has been checked against real learners.

### 14. The capability check covers knowledge, not behaviour

ARC-Challenge, MMLU stem and GSM8K test whether the model still knows things. They cannot see a loss
of instruction-following, formatting, multi-turn coherence, or writing quality — and the
multiple-choice ones are scored by option loglikelihood specifically to *avoid* depending on format
compliance, which means a policy that had lost the ability to follow a formatting instruction would
still score 70.8% on ARC. The null is real but narrower than "capability is untouched" suggests.

---

## Tier 4 — smaller, still worth fixing

15. **Only the final checkpoint was evaluated.** Saves exist every 10 steps and the paper's own
    recommendation is to test them; until that happens, "200 steps" is an arbitrary stopping point
    that may be past peak quality.
16. **The length band is one person's taste.** The 18–40 word target comes from 1131 length
    judgements by the same single rater who provides all the human evaluation. It is treated as
    ground truth about what students need.
17. **Reward hacking was tested only for one known attack.** Injecting the gold answer is a good
    test and the hardening result (93% → 3%) is solid. But it tests a hypothesis someone had in
    advance. Nothing searches for exploits the policy found on its own — no inspection of the
    highest-scoring turns for pathology.
18. **Small-sample statistics are used freely.** Spearman-Brown corrections, agreement ceilings and
    effective rank are all computed on samples of 17–600, and their uncertainty is not propagated
    into the comparisons that rely on them.

---

## What is actually solid

For calibration, because a document like this is misleading if read as a verdict:

- The negative result on outcome-based rewards is convincing and well evidenced ($r=-0.012$ against
  $+0.291$), and it is a genuinely useful thing to have established.
- The surface-feature control on the probe is the right control and it is what makes `targeted` at
  0.85 against 0.36 believable.
- The `leak` hardening is real and the fitting script's refusal to emit a head that fails the attack
  gate is good practice.
- The capability check is properly paired and honestly bounded, and the two harness bugs found on the
  way to it are documented rather than quietly fixed.
- The `leak` decoupling failures are reported as failures with a correct diagnosis, which is rarer
  than it should be.

## If only three things get done

1. **Sample a length-instructed base model** and put it in the blind pool (item 2). Cheapest, and it
   is the one that could invalidate the headline.
2. **Report the headroom table and soften the ceiling claim** (item 1). Costs nothing. The paper
   currently attributes the plateau to the reward model's expressible range, and the probe was
   still offering 0.578 when the runs stopped.
3. **Finish the pairwise preferences** (item 4). The only rubric-independent evidence, and the pairs
   already exist.
