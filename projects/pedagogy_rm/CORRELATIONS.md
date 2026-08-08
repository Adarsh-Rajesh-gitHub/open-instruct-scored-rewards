# The dimensions are not measuring what the rubric says

What the reward's components correlate with, why rewriting the rubric does not fix it, and what to do
instead. Written 2026-08-06, while arm H is still running.

Everything below is recomputed from `data/hidden.npz`, `data/head5.npz`, `data/labels/*.json` and
`data/pilot/`, not quoted from the report.

---

## 1. The finding

### The reward correlates −0.60 with turn length

Running `head5.npz` over all 600 labelled turns, and the 7-rater consensus labels over the same
turns. All values on the raw 1–3 scale, `leak` **not** sign-flipped, so 3 = gives the answer away.

| | leak | targeted | actionable | elicits | correct | log(words) |
|---|---|---|---|---|---|---|
| **leak** | 1.00 | 0.39 | −0.42 | −0.52 | −0.56 | **+0.64** |
| **targeted** | 0.39 | 1.00 | 0.38 | 0.34 | −0.17 | **+0.03** |
| **actionable** | −0.42 | 0.38 | 1.00 | **0.96** | 0.27 | −0.42 |
| **elicits** | −0.52 | 0.34 | **0.96** | 1.00 | 0.30 | −0.52 |
| **correct** | −0.56 | −0.17 | 0.27 | 0.30 | 1.00 | −0.58 |

Now apply the signs the reward actually uses. `leak` is negated, so its +0.64 becomes −0.64 and it
joins the other three pointing the same way:

| signed as rewarded | probe | labels |
|---|---|---|
| `leak` | −0.64 | −0.53 |
| `correct` | −0.58 | −0.43 |
| `elicits` | −0.52 | −0.50 |
| `actionable` | −0.42 | −0.40 |
| `targeted` | **+0.03** | **+0.02** |
| **composite reward** | **−0.60** | **−0.57** |

**Four of five dimensions push toward shorter turns.** Averaging them does not average away the
length signal, it accumulates it. The reward is mostly a brevity detector.

### The probe amplifies this but did not create it

| | probe | labels |
|---|---|---|
| mean \|r\| among the five | 0.43 | 0.37 |
| max \|r\| | 0.96 | 0.95 |
| first principal component | 51% | 47% |
| effective rank | 3.07 / 5 | 3.31 / 5 |
| composite vs log(words) | −0.60 | −0.57 |

Every probe correlation matches the labels in sign and exceeds it by 0.01–0.15. So this is a property
of **the ratings**, not an artefact of reading them out of activations. The probe's own contribution is
a mild sharpening, largest on length (`leak` +0.11, `correct` +0.15).

Two consequences worth separating:

- **Redundancy.** `actionable` and `elicits` correlate at 0.96. That is one quality with two names,
  counted twice in a five-way average.
- **Length.** Independent of the redundancy, the composite is a length penalty. This is the one that
  drives behaviour.

### It explains the whole project history

Not a new observation alongside the others — the mechanism underneath them:

- **Arm A collapsed to 12 words.** With no length term, a −0.60 reward has its optimum at "as short as
  possible". Nothing was wrong with the optimiser.
- **A length band had to be bolted on.** It is the only term pushing back against four that pull one
  way.
- **The length term supplies ~70% of the measured gain.** It is doing the work because it is the only
  component not already saturated by getting shorter.
- **The probe dimensions captured only 40% of their headroom** (0.230 of 0.578) while the length term
  took 97% of an almost identical 0.604. The unused probe headroom sits *behind* the collapse the band
  blocks.
- **`targeted` moved least of all — 21% of its room.** It is also the only length-neutral dimension.

### Arm H confirms it directly

Arm H is arm E with the length term removed and nothing else changed. At step 78 of 200:

| | arm H | arm E (final) |
|---|---|---|
| words | **17.9** | 27.9 |
| 5-dim mean | **1.886** | 1.852 |
| steps | 78 | 200 |
| KL drift | 0.36 | 0.25 |

**Arm H passed arm E's final probe score in 78 steps by getting shorter**, and 17.9 words is already
below the 21-word floor of the human-acceptable range measured over 1131 judgements. The band was
holding the policy off a higher-scoring path, exactly as the correlation predicts.

*Caveat, and it runs against the conclusion:* arm H has drifted **more** than arm E at every matched
step (0.36 against 0.16 at step 70). I had predicted the opposite — dropping the length term removes
within-group reward variance, which should shrink the effective step. It did not, so arm H's probe
advantage is partly bought with distance travelled rather than being free. The clean version needs the
two arms matched on drift, not on learning rate.

---

## 2. Rewriting the rubric does not fix it

V2 was built partly for this: `guidance` ("does the turn supply anything usable") was meant to be
length-**positive** and offset the others. Nine raters scored 44 turns on all six V2 dimensions, so it
is testable.

| | V1 (n=600) | V2, all 44 | V2, **real turns only** (n=20) |
|---|---|---|---|
| mean \|r\| | 0.37 | 0.22 | 0.28 |
| max \|r\| | **0.95** | 0.65 | 0.70 |
| first PC | 47% | 36% | 40% |
| effective rank | 3.31 / 5 | 4.88 / 6 | **4.53 / 6** |
| composite vs log(words) | **−0.57** | −0.37 | **−0.62** |

**Read the third column.** 24 of the 44 V2-labelled turns are manufactured negatives, each corrupted
along a single dimension — which decorrelates the dimensions *by construction* and says nothing about
the rubric. The middle column's encouraging −0.37 is that artefact. Excluding them leaves 20 real
turns.

**V2 fixes redundancy.** The 0.96 duplicate is gone (max 0.70) and effective rank rises from 3.31 of 5
to 4.53 of 6 even on real turns. V2 measures about four and a half things where V1 measured about
three. That is a real gain and worth adopting on its own.

**V2 does not fix length.** −0.62 against V1's −0.57. Per dimension on real turns:

| V2 dimension | signed length coupling |
|---|---|
| `correct` | −0.66 |
| `verdict` | −0.53 |
| `hands_over` | −0.51 |
| `leak` | −0.26 |
| `guidance` | **−0.12** |
| `locates` | **−0.06** |

The two dimensions designed to be length-neutral are neutral. The other four swamp them. And
`guidance`, the intended length-*positive* counterweight, came out at −0.12 — because "supply
something usable" can be satisfied in eight words.

At n=20 the interval on −0.62 is about [−0.84, −0.24], so this cannot show V2 is *worse*. It rules out
the hope that it helps.

### Why wording is the wrong lever

Decorrelating the dimensions **from each other** and decorrelating them **from length** are different
problems. V2 solves the first because that one is about overlapping definitions. The second is not
about definitions: raters score shorter turns higher on most pedagogical qualities under either
rubric, so the coupling is in their judgements before any wording is chosen. Any head fitted on this
corpus inherits it. A third rubric would move effective rank and leave −0.6 where it is.

---

## 3. What to do

### Settled since this was written: the probe is not just a length detector

The obvious worry from a −0.60 coupling is that the probe measures length and little else, so any
trained policy is only a shorter policy. That has now been tested and the answer is no. Sampling base
with `Keep your reply between 18 and 40 words.` appended, on the same held-out questions:

| policy | words (median) | 5-dim mean |
|---|---|---|
| plain base | 35 | 1.547 |
| base prompted to 18--40 | **28** | 1.475 |
| arm E (trained) | **27.2** | **1.879** |

The prompt did shorten the turns — median 35 to 28, matching arm E's 27.2 — and the probe score went
slightly *down*. **At matched length the trained policy scores +0.40 above a prompted base.** So there
is real length-independent signal in the probe, and roughly a third of arm E's 5-dimension level is
something instruction alone does not reach.

This changes what the coupling means. It is not evidence that the reward is worthless; it is evidence
that the reward cannot be trusted to set *length*, which is a narrower and more manageable problem —
and one the explicit band already solves. Treat the probe as the quality term and the band as the
length term, and stop expecting either to do the other's job.

(Caveat: the prompted run drew the first 100 eval prompts and arm E's held-out evaluation covers all
200, so the question sets overlap without being identical. The 0.40 gap is well beyond the 0.32
within-sample standard deviation, but an exact rerun on identical prompts would be cleaner.)

### Do first: find out whether the coupling is confound or preference

This is the fork everything else depends on, and nobody has established which side we are on.

- **Confound:** in this corpus, longer turns *happen* to be worse — they over-explain, they leak, they
  ramble — so raters correctly score them lower and length rides along as a proxy. Fixable with better
  data.
- **Preference:** shorter really is better per unit of content, and −0.6 is a true property of good
  tutoring. Then the reward is right and only the floor needs enforcing.

**The experiment.** Take ~60 existing turns. For each, generate two length variants that hold content
fixed — one compressed, one expanded — changing only wording density, not what is said. Rate all
180 blind on the V2 dimensions.

- If ratings stay flat across length variants, the coupling was confound: length was standing in for
  content differences, and a length-balanced corpus removes it.
- If ratings still fall with length at fixed content, it is preference. Stop trying to remove it and
  keep an explicit floor instead.

Cost: one generation job plus 180 ratings. Decisive either way, and it is the cheapest thing here.

### Do regardless

1. **Keep the explicit length term.** Arm H is the argument: 78 steps to fall below the human floor.
   The band is load-bearing, not a hack. But state its share honestly — 2.0 of a 4.2 maximum is 48% of
   the objective by design, and the paper never said so.
2. **Fit a V2 head and use it.** The redundancy fix is real and independent of the length question.
   Costs one labelling round plus one extraction job. Expect no change in length behaviour.
3. **Match arm H on drift.** It drifted 0.36 against arm E's 0.25, so rerun it at a lower learning
   rate targeting 0.25. Without that, the headroom comparison is not clean.

### The one design change available with no new data

**Build the reward from the length-neutral dimensions only, plus the explicit length term.**

The length-neutral dimensions are `targeted` (+0.03 in V1), and in V2 `guidance` (−0.12) and `locates`
(−0.06). Everything else is substantially a length proxy. A reward of
`mean(locates, guidance) + w x length_fit` has a property the current one lacks: **length enters
through exactly one term, which someone wrote down and can tune, rather than through four that were
not intended to carry it.**

There is a reason to think this is not merely a subtraction. `targeted`/`locates` is precisely where
the activation probe beats the surface-feature control by the widest margin — 0.85 against 0.36. The
dimensions that are not length proxies are the ones where reading activations adds the most, because
the others were partly recoverable from word count all along. The case for the probe is strongest on
exactly the subset this proposal keeps.

The cost is a thinner reward: two dimensions instead of five or six, and `guidance` is also the
weakest on rater agreement (κ 0.51). Worth measuring against the current reward rather than assumed
better.

### Do not do

- **Another rubric rewrite aimed at length.** V2 was that attempt and the measurement above is the
  result.
- **Residualising length out of the hidden states.** Tried twice. Both failed out of distribution
  because they were estimated in distribution, and a corpus where length and quality are confounded
  is why. This becomes viable only *after* the corpus is fixed, not instead of fixing it.

---

## Open questions

- **Does the coupling hold on policy-generated text?** All of the above is measured on base-model
  turns. The `leak` head's coupling was +0.44 in distribution and +0.70 on policy text, so these
  numbers are probably understated for the distribution training actually runs on.
- **Why did arm H sit at 20–21 words from step 40 to 70, then resume falling?** A shelf then a further
  descent suggests two different gradients — plausibly the four length-coupled dimensions saturating,
  then something else taking over. Worth identifying before extending any run.
- **Is `correct` at −0.66 telling us that longer turns are genuinely more often wrong?** That would be
  a real pedagogical finding rather than a measurement artefact, and it is separable: check whether
  human `correct` ratings fall with length at fixed content in the experiment above.
