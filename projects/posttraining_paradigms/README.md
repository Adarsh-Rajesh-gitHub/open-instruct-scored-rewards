# Three bets for post-training beyond vanilla GRPO

Research memo, 2026-08-06.

## Executive call

Current RL/GRPO post-training is good at moving probability mass toward rewards
the model can already reach. It is much less reliable at:

1. improving one capability without quietly deleting another;
2. expanding the set of problems the model can solve, rather than only raising
   `pass@1`; and
3. producing coherent behavior over an interaction instead of a good isolated
   answer.

The three bets below attack different failure modes:

- **Transactional GRPO:** treat every policy update like a code change that must
  pass behavioral canaries before it is committed.
- **Counterfactual Student-State GRPO:** use paired simulator forks to estimate
  delayed transfer under tutor actions, then validate that signal on learners.
- **Conversation-Diamond GRPO:** train equivalent multi-turn paths to end in the
  same correct state, while requiring the model to react when the user makes a
  materially relevant correction.

Recommendation: run each as a short falsification sprint, not as a platform
project. Start with Transactional GRPO because it has the cleanest measurement
and the broadest payoff. Run Student-State GRPO second because it has the
largest upside and the largest simulator-validity risk. Run Conversation-Diamond
GRPO if a deterministic task family with explicit invariances can be built.

These are research hypotheses, not priority claims. The search below is
selective, not a formal novelty review.

## What the literature says

Several results change what a credible proposal must measure:

- **Capability forgetting and potential collapse are different.** The first is
  regression on unrelated or previously mastered tasks. The second is a rise in
  `pass@1` while rare valid strategies or high-`k` coverage disappear. They can
  move independently and must be reported separately.
- RLVR often raises `pass@1` while failing to exceed, or even shrinking, the
  base model's high-`k` capability boundary
  ([Yue et al., 2025](https://arxiv.org/abs/2504.13837),
  [Scalpel vs. Hammer](https://arxiv.org/abs/2507.10616)).
- Forgetting is not adequately measured by aggregate accuracy. Individual
  prompts enter and leave the correct set throughout training
  ([ReMind](https://arxiv.org/abs/2606.03087)), and rare correct modes can vanish
  while common modes improve
  ([F-GRPO](https://arxiv.org/abs/2602.06717),
  [Pass@k Inversion](https://arxiv.org/abs/2607.20543)).
- A global KL penalty is a weak proxy for useful behavior. Recent work instead
  uses dynamic replay, support constraints, gradient conflict resolution, or
  robustness to future fine-tuning
  ([RECAP](https://arxiv.org/abs/2510.21978),
  [APO](https://arxiv.org/abs/2602.05717),
  [PCR](https://arxiv.org/abs/2602.06453),
  [FRPO](https://arxiv.org/abs/2602.08813),
  [SimKO](https://arxiv.org/abs/2510.14807),
  [correctness-conditioned KL](https://arxiv.org/abs/2608.01743)).
- Multi-turn quality needs state and credit assignment. Terminal trajectory
  rewards blur which turn helped; branching and turn-level methods recover
  better signals
  ([MT-GRPO](https://arxiv.org/abs/2505.11821),
  [Conversation Forests](https://arxiv.org/abs/2507.04099)).
- Tutoring work is moving from next-answer correctness toward explicit cognitive
  state and long-term outcomes
  ([UCO](https://arxiv.org/abs/2511.08873),
  [PEARL](https://arxiv.org/abs/2605.29582)).
- Consistency rewards over paraphrases already exist. Merely grouping equivalent
  prompts and rewarding similar answers is not a new paradigm
  ([PS-GRPO](https://arxiv.org/abs/2510.04392),
  [multi-turn persona consistency](https://arxiv.org/abs/2511.00222)).
- Evaluation-based stopping and checkpoint preservation already exist at the
  scheduler level ([EvalStop](https://arxiv.org/abs/2606.04145)).
- Multi-turn metamorphic testing and canonical-versus-incremental path training
  are direct precedents for conversation-path consistency
  ([MORTAR](https://arxiv.org/abs/2412.15557),
  [CCOPD](https://arxiv.org/abs/2605.30251)).
- Prompted student simulators are not reliable stand-ins for real students;
  fine-tuning helps but remains limited
  ([Scarlatos et al., 2026](https://aclanthology.org/2026.acl-long.1960/)).

That rules out three easy but weak ideas: "add more KL," "replay old data," and
"reward similar answers."

---

## Bet 1: Transactional GRPO

### One-line thesis

Replace a likelihood-space trust region with a **behavioral commit protocol**:
propose an RL update, test the shadow policy on rotating capability canaries,
and commit, shrink, repair, or reject the update based on measured behavior.

### Hypothesized advantage over current GRPO

Normal GRPO asks whether an update is locally acceptable under clipping or KL.
Transactional GRPO asks whether the resulting model still passes its behavioral
contract. The unit of control is the optimizer transaction, not a token-level
distance from a reference model.

This matters because two policies can be close in average KL and differ on a
rare capability that users care about. Conversely, they can be far in KL while
remaining behaviorally equivalent on protected capabilities.

### Mechanism

Maintain three disjoint streams:

1. **Acquisition batch:** the normal on-policy GRPO data.
2. **Rotating canary bank:** small, stratified capability slices used to decide
   whether a candidate update is safe.
3. **Sealed audit bank:** never used for update decisions; it tests whether the
   canaries generalize.

For every `N` optimizer steps:

1. Compute the proposed GRPO update and apply it to a shadow policy
   `theta_candidate`.
2. Evaluate current and candidate policies on identical prompts with repeated
   rollouts. Reuse RNG noise only if a pilot establishes positive paired
   covariance; identical seeds alone do not couple diverged autoregressive
   paths.
3. Estimate prompt-level paired differences with a cluster bootstrap or an
   anytime-valid confidence sequence. Pre-register non-inferiority margins and
   limit adaptive queries to each canary slice; refresh a slice when its query
   budget is exhausted.
4. Commit only if the acquisition gain is positive and every protected
   capability's lower confidence bound is above its allowed regression budget.
5. If the transaction fails, backtrack the step size. If backtracking still
   fails, generate fresh rollouts only for the failed canary families and
   project the acquisition gradient against their recovery gradients. If no
   feasible update remains, reject it.

A simple acceptance rule for capability family `j` is:

```text
LCB[metric_j(theta_candidate) - metric_j(theta_current)] >= -epsilon_j
```

The gate is a sequential statistical test, not a dashboard heuristic. Repeated
inspection invalidates ordinary fixed-sample confidence intervals. The canary
bank therefore needs power analysis, multiplicity control, query budgets, and a
sealed audit set
([paired-evaluation diagnostics](https://arxiv.org/abs/2605.30315)).

### What is actually new

The closest methods protect behavior *inside the objective*:

- KL protects the current-task distribution.
- RECAP and ReMind resample or replay at-risk prompts.
- APO and prompt-balanced anchoring protect reference support.
- PCR resolves estimated stability/plasticity gradient conflict.
- FRPO optimizes robustness to a neighborhood of future policies.

Proposal, evaluation, line search, and rollback are not new in RL; TRPO, CPO,
and safe policy-improvement methods are clear ancestors. EvalStop also preserves
the best checkpoint after downstream metrics decline.

The candidate contribution is narrower: **multi-capability non-inferiority
gating before a committed LLM-RL update, coupled to targeted recovery
gradients**. This is empirical safe policy improvement specialized to stochastic
LLM evaluation, not a claim to have invented update acceptance.

The novelty claim fails if prior work already performs paired, multi-capability,
pre-commit behavioral acceptance with rollback during LLM RL. That should be
the first literature-review question.

### Minimum experiment

- Model: a 1B-3B instruct model with math RLVR as the acquisition task.
- Protected canaries: instruction following, short factual QA, code, safety
  refusal boundaries, and held-out math.
- Budget: choose prompts per family from a prospective minimum-detectable-effect
  power analysis. Reserve high-`k` evaluation for checkpoint and sealed-audit
  intervals.
- Baselines: vanilla GRPO, tuned KL-GRPO, ReMind, one support-preserving method,
  EvalStop, and compute-matched best-feasible checkpoint selection after a
  normal run.
- Ablations: gate only, targeted repair only, and gate plus repair.
- Primary metrics:
  - acquisition `pass@1`;
  - protected-task worst-case regression;
  - base-to-final correct-set loss;
  - verifier-valid CoT `pass@1` and `pass@64`, not answer-only lucky hits;
  - semantic strategy coverage among verified solutions;
  - few-step adaptation speed on a held-out domain as a measure of retained
    functional plasticity;
  - wall-clock overhead and rejected-update rate.

The decisive result is not a higher mean score. It is a better acquisition gain
at the same worst-family regression budget on the sealed audit bank.

### Open-Instruct fit

The repository already has:

- checkpointable GRPO actors and configurable reference policies;
- local evaluation with configurable `pass@k`;
- verifier and group-scorer plugins; and
- reward logging that can expose zero-advantage and per-dimension metrics.

The first prototype can run the canary gate every checkpoint interval outside
the inner optimizer. A real version needs a hook around optimizer commit plus a
cheap shadow-weight or reversible-update path. This is therefore a trainer
change, not only a `projects/.../plugin.py`.

### Failure modes and kill criteria

- **Canary overfitting:** sealed audit regression remains unchanged.
- **Noisy vetoes:** paired confidence intervals still reject more than 30% of
  steps without predicting audit harm.
- **Cost:** less than half the training throughput remains after amortization.
- **Gradient deadlock:** protected families leave no useful acquisition update.
- **False safety:** capability loss moves to unmeasured prompt forms.

Kill the idea if canary decisions do not predict sealed-audit changes better
than simple KL drift or gradient norm.

---

## Bet 2: Counterfactual Student-State GRPO

### One-line thesis

Use paired simulator forks to obtain a lower-variance estimate of delayed
transfer under tutor actions, and test whether that simulator-derived signal
predicts randomized human learning effects.

### Hypothesized advantage over current GRPO

Most tutor rewards score an utterance, the next student answer, or the final
dialogue. All three confound tutoring quality with the student's initial state,
the simulator's sampling noise, and answer leakage.

The proposed unit of optimization is a state transition:

```text
(student state before) -- tutor action --> (student state after)
```

The control branch estimates the policy effect for gating and evaluation. The
within-group reward remains delayed transfer unless a nonlinear improvement
event is used, because any shared additive control baseline cancels in GRPO.

### Mechanism

Give the student environment an explicit hidden state such as:

- mastered skills;
- active misconceptions;
- confidence or uncertainty;
- tendency to guess; and
- memory of prior hints.

At each tutor decision:

1. Snapshot the same student state and, where the simulator supports principled
   coupling, its exogenous randomness.
2. Sample `G` candidate tutor actions from the policy.
3. Fork the environment once per action plus a control action such as silence,
   a generic encouragement, or the current production tutor.
4. Continue each branch for a fixed horizon.
5. Remove the tutor and administer a fresh isomorphic problem.
6. Score each action by delayed transfer minus leakage and unnecessary-help
   penalties:

```text
reward_i =
  post_transfer_i
  - leak_penalty_i
  - unnecessary_help_i
```

7. Use the paired control outcome for variance reduction, non-inferiority
   gating, and reporting. Under sibling-normalized GRPO, subtracting the same
   pretest and control score from every sibling cancels and does not define a
   new optimization objective. If improvement over control must change the
   objective, use a pre-registered nonlinear event such as
   `post_transfer_i - post_transfer_control > delta`.
8. Compute sibling-relative advantages only among branches from the same
   student snapshot.

Add a state-belief task: before acting, the tutor predicts the student's hidden
state in a machine-readable side channel. Reward calibration against the
environment state, but never let the tutor read that state. At inference, strip
the side channel or replace it with an auxiliary head.

The transfer problem is load-bearing. Without it, the easiest policy is to hand
the student the answer.

### What is actually new

UCO rewards estimated cognitive progress and productive struggle. PEARL builds
an explicit cognitive student simulator. Existing branching methods compare
downstream outcomes among sibling conversations.

Student-state RL, POMDP tutoring, pre/post learning gains, and branching are not
new individually. The candidate contribution is:

- paired delayed-transfer evaluation from identical simulator state;
- joint state-belief learning with no-head, shuffled-label, and oracle-state
  controls;
- cross-simulator validation; and
- a pre-registered randomized human treatment effect.

Simulator forks identify effects only inside the specified simulator. The
research result must be that this signal predicts learning outside that
simulator better than raw cognitive-progress or terminal-answer scores.

### Minimum experiment

- Domain: single-skill and two-skill arithmetic misconceptions where the
  simulator state can be exact.
- Student: a controlled state machine that verbalizes through a frozen LLM,
  followed by a learned student simulator only after the method works.
- Tutor: a 1B-3B policy, three turns, `G=4` candidate branches plus control.
- Baselines: terminal-answer GRPO, UCO-style progress reward, unbranched
  multi-turn GRPO, and Conversation-Forest-style sibling credit.
- Held-out axes:
  - unseen numbers and wording;
  - unseen misconception combinations;
  - longer delay before transfer;
  - a separately trained student simulator;
  - a pre-registered randomized human evaluation with delayed isomorphic
    transfer, clustered by student and powered for the treatment effect.
- Primary metrics:
  - independent post-test gain;
  - retention after a delay;
  - leak rate;
  - state-belief calibration;
  - policy effect over the control tutor; and
  - transfer across student simulators.

### Open-Instruct fit

This is a natural project plugin:

- `PartnerModelEnv` already supports a frozen model as a multi-turn environment.
- A custom `Director` can own the hidden student state and constrain valid
  student behaviors.
- Environment `info` can record state transitions and policy-only turns.
- A `GroupScorer` can compute sibling-relative delayed-transfer rewards.

True branching from an intermediate snapshot is not currently a first-class
rollout topology. The MVP can recreate forks from deterministic serialized
state; a scalable version will need explicit environment snapshot/restore.

### Failure modes and kill criteria

- **Simulator laundering:** the tutor exploits simulator quirks that do not
  transfer to another simulator or humans. Prompted model students are
  specifically disallowed as the only validation environment.
- **State fiction:** the explicit state does not predict future student actions.
- **Control contamination:** the chosen control is too weak, making every tutor
  look causal.
- **Branch cost:** `G x horizon` overwhelms the value of the lower-variance
  estimate.
- **Belief leakage:** the state side channel teaches the policy to emit labels
  without using them to choose better actions.

Do not start RL until the student environment passes intervention tests: changing
one hidden misconception while holding the prompt fixed must change future
errors in the intended way. Kill the model-based phase if gains do not transfer
to a separately built student.

---

## Bet 3: Conversation-Diamond GRPO

### One-line thesis

Train on **commuting conversation diagrams**: different but semantically
equivalent sequences of turns should reach the same correct commitments, while
non-equivalent corrections must produce the appropriate change.

### Hypothesized advantage over current GRPO

Single-turn consistency asks, "Do paraphrases get similar answers?" Multi-turn
consistency should ask, "Does the model maintain the same world state when the
conversation takes an equivalent path?"

For example, these paths may be equivalent:

```text
Path A: give constraint X -> ask question -> give irrelevant interruption
Path B: irrelevant interruption -> give constraint X -> ask paraphrased question
```

These paths must not be treated as equivalent:

```text
Path A: budget is $100 -> recommend an option
Path B: budget is corrected to $50 -> revise the recommendation
```

A model that always repeats itself is consistent but wrong. The training set
must specify both invariances and required sensitivities.

### Mechanism

Build a generator of conversation diamonds. Each item contains:

- an underlying task state;
- two or more valid paths through that state;
- a typed transformation system with explicit composition rules, identity
  transformations, state-transition semantics, and enumerated pairs that should
  or should not commute;
- a deterministic verifier for the final answer or action; and
- an extractor for explicit commitments made along the path.

For each GRPO group, the environment samples different paths from the same
underlying item. Never assign one group-level diamond score to every sibling:
that produces zero GRPO advantage. Give each path an oracle-based endpoint score
plus its leave-one-out marginal contribution to relation satisfaction. The
MVP assigns that scalar to the whole path; a later trainer change can assign
span-specific credit where commitments are created or revised. The scorer
computes:

1. **Endpoint correctness:** every path must solve the task.
2. **Invariant consistency:** invariant paths must agree on facts, constraints,
   and selected actions, not necessarily wording.
3. **Correction responsiveness:** relevant edits must change exactly the
   commitments they invalidate.
4. **Recovery:** a distractor or interruption must not erase prior constraints.
5. **Non-collapse:** preserve multiple valid explanations or plans when the task
   genuinely permits them.

Apply consistency bonuses only after correctness gating. Otherwise the model can
earn reward by being uniformly wrong.

A stronger variant places the consistency bonus in **reward escrow**: a path
earns ordinary task reward immediately, but receives the relation bonus only
after passing delayed transformations hidden from the rollout policy, such as
constraint reordering, equivalent specifications, variable renaming, or
generated counterexamples. Rotate these transformations so they do not become
another stable public verifier.

A useful headline metric is:

```text
diamond success =
  P(all endpoints correct
    AND invariant commitments agree
    AND corrected commitments change appropriately)
```

### What is actually new

PS-GRPO groups paraphrased prompts and rewards similar content. Persona work
rewards prompt-to-line and line-to-line consistency. Conversation Forests and
Tree-GRPO branch trajectories for credit assignment. More direct precedents are
MORTAR's multi-turn metamorphic relations, tau-bench's deterministic final-state
equivalence, and CCOPD's equivalent-evidence path training.

The candidate contribution is joint RL over positive relations that should
commute and negative relations that must change specified commitments, using
deterministic state-transition oracles and per-path marginal credit. We do not
claim novelty for path equivalence, final-state verification, or branching
alone.

The idea is not novel if the final implementation reduces to embedding
similarity between paraphrased answers. The explicit state, transformation
labels, correctness gate, and correction test are the contribution.

### Minimum experiment

Start where state is verifiable:

- multi-turn instruction following with reordered constraints;
- database or calendar updates;
- shopping under changing hard constraints; or
- code debugging where the user adds and retracts requirements.

Use 500-2,000 underlying items and generate diamonds programmatically. Hold out
transformation compositions, not random conversations.

Baselines:

- standard GRPO on one path;
- mixed-path GRPO without a consistency term;
- PS-GRPO-style endpoint similarity;
- line-to-line consistency reward;
- MORTAR-style metamorphic training/evaluation;
- CCOPD and a canonical-state supervised oracle;
- a larger supervised model.

Measure:

- diamond success;
- endpoint accuracy;
- contradiction rate;
- recovery after distractors;
- correct revision after material edits;
- diversity among valid explanations;
- single-turn capability regression; and
- performance as path length grows.

Evaluate on transformations from an independently authored generator, including
MORTAR-, SEQUOR/EvolIF-, or tau-bench-style state transitions. Holding out only
compositions from the training generator does not exclude generator artifacts.

### Open-Instruct fit

The existing stack is unusually well suited to the MVP:

- `TextRLEnvironment` supports multi-turn interactions.
- A `Director` can choose the next transformation before generating user text.
- Environment `info` can retain the latent task state and transformation IDs.
- `GroupScorer` already sees every completion in a prompt group and can compare
  commitments across paths.
- The anchor interface can evaluate unseen transformation compositions that are
  absent from the reward.

One limitation is that current GRPO groups originate from one prompt. Keep the
opening prompt fixed and sample the path inside the environment so siblings
share the same underlying task. Another limitation is credit granularity:
`GroupScorer` returns one scalar per trajectory. True turn-level credit requires
retaining a reward vector and applying span-specific advantages in
`data_loader.py` / the GRPO trainer.

### Failure modes and kill criteria

- **Conformity collapse:** outputs become bland or identical.
- **Extractor hacking:** the model games commitment parsing.
- **Bad equivalence labels:** supposedly invariant paths change the task.
- **Judge leakage:** a language-model judge rewards style instead of state.
- **No transfer:** gains vanish on unseen compositions or longer dialogues.

Kill the approach if deterministic state checks do not outperform generic
semantic-similarity rewards on held-out compositions.

---

## Portfolio plan

### Sprint 0: common instrumentation

Before training anything:

- record per-item correct-set acquisition and regression;
- evaluate both `pass@1` and at least `pass@64`;
- keep prompt-paired base/current rollouts with repeated generations;
- report the worst capability family, not only the mean;
- define a held-out anchor before defining the reward; and
- hold out causal axes: capability family, student misconception, or
  transformation composition.

### Two-week falsification order

1. **Transactional GRPO**
   - Days 1-3: determine whether tiny paired canaries predict checkpoint
     regressions.
   - Days 4-7: add checkpoint-level accept/reject and backtracking.
   - Days 8-10: compare against KL and ReMind on the same compute budget.
   - Stop unless sealed-audit regression improves.

2. **Counterfactual Student-State GRPO**
   - First build a deterministic student state machine.
   - Prove delayed transfer separates helpful hints from answer leakage.
   - Only then replace verbalization with a model student.
   - Stop unless gains transfer across student implementations.

3. **Conversation-Diamond GRPO**
   - Generate a small deterministic benchmark before touching the trainer.
   - Test whether the base model fails more on composition than on individual
     transformations.
   - Train only if that gap is large enough to measure.
   - Stop unless held-out transformation compositions improve.

## Bottom line

The highest-leverage shift is to stop treating "the reward went up" as the
definition of progress.

- Transactional GRPO makes **retention an update-level acceptance criterion**.
- Student-State GRPO makes **delayed transfer, not the next answer, the reward**.
- Conversation-Diamond GRPO makes **path-coherent state evolution the unit of
  consistency**.

Each proposal has a cheap test that can kill it before a large run. That is a
feature, not a limitation.
