# Persistent mastery-graph tutoring

## Research question

Can a small tutor update a persistent per-concept mastery graph from student
evidence, and does that graph improve prerequisite selection and delayed
transfer across questions?

The claim is deliberately narrow. This project does not attempt to model human
cognition or reproduce PEARL at smaller scale. It tests whether a small model
can maintain a compact record of demonstrated mastery, uncertainty, and missing
prerequisites in a partially observed tutoring environment.

## Pivot after initial falsification

Three 150–180 example ablations found:

- raw wrong-answer state made tutoring worse;
- abstract mistaken-relation state did not beat a counterfactual relation or
  concept-only scaffold; and
- persistent mistaken-relation memory did not beat concept-only memory.

Therefore, a static misconception label is no longer the primary state. The
state is now a temporal graph whose concept nodes record:

- mastery: `unknown`, `partial`, or `mastered`;
- uncertainty or confidence;
- the evidence supporting the current value;
- the last update turn; and
- an optional misconception relation only when directly supported.

All hidden-state misconception probing below is conditional on first showing
that this mastery graph has oracle decision value, can be updated without
corrupting unrelated nodes, and improves held-out transfer.

## Current empirical status

Real-human SPHERE responses now establish two positive results:

- per-domain mastery improves held-out response prediction from 0.753 to 0.833
  AUC over item difficulty plus global ability; and
- the graph identifies the student's held-out weakest domain 53.7% of the time,
  versus 14.3% chance, with domain-rank correlation 0.621.

State updates are useful after only one response per domain and improve
monotonically through 12 responses per domain.

The stronger broad-transfer claim failed. Mastery from one related instrument
usually did not beat mastery from an unrelated instrument, except for the
FCI–RRMCS mechanics/rotation pair. Consequently:

1. begin with explicit Beta-Bernoulli mastery nodes updated from observed
   correctness;
2. use the state first for structured curriculum selection—what skill to
   practice next;
3. do not train a hidden-state misconception probe;
4. require fine-grained isomorphic skill mappings for transfer; and
5. require a causal learning gain over random and global-ability curricula
   before using GRPO.

## Teacher-first integration decision

The primary LLM integration is now the **teacher policy**, not the student
simulator. The full design is in
`projects/posttraining_paradigms/mastery_graph_architecture.md`.

Separate three claims:

1. A probabilistic estimator predicts a real student's future performance from
   prior evidence.
2. An LLM teacher uses that state to select and generate the next intervention.
3. A synthetic student is only a controlled test fixture; state-conditioned
   prompt compliance is not evidence of human learning dynamics.

The environment owns state updates. The teacher receives a bounded graph
neighborhood and emits a structured decision:

```json
{
  "target_node": "identify_interaction_pairs_text",
  "action": "contrast_case",
  "item_id": "n3_bank_q14",
  "stop": false
}
```

A separate generation step realizes this decision as tutoring language. This
distinguishes improved pedagogical selection from improved wording.

Because data is limited, the MVP uses:

- 20–50 ready, text-only ESTELA isomorphic banks;
- one initial mastery leaf per bank;
- the bank's unit as its parent collection;
- weighted Beta-Bernoulli beliefs with hierarchical priors; and
- no prerequisite-state propagation until transfer evidence validates an edge.

The first frozen-teacher ablation compares latest-turn, full-history,
concept-only, correct-graph, student-shuffled-graph, node-permuted-graph,
weakest-node, information-gain, and oracle conditions. It first scores
structured action sensitivity, then tests the generated intervention on a
sealed isomorphic item.

Stop if the correct graph does not change decisions, fails to beat
transcript-only, or only matches the best deterministic scheduler. Use SFT only
when the frozen teacher understands the state but makes systematic action
errors. Use GRPO only after graph-SFT leaves meaningful regret and independent
transfer rewards are reliable.

## Core design rule

The hidden-state probe is a diagnostic, not the reward.

The environment owns the true student state and computes delayed transfer
exactly. This avoids using one learned probe both to define success and to
conclude that success occurred.

The hidden state is not an additional source of student information. It is a
deterministic transformation of the question and response. Its possible value
is as a compact, action-usable representation of evidence already present in
the transcript. The project should not claim that a probe discovers information
unavailable in the text.

## Phase 1: Build an exact student environment

Create a new project at `projects/student_state_tutor/`.

### State space

Start with four to eight controlled states:

- specific misconception types;
- mastered versus unmastered target skill;
- correct but uncertain;
- guessing without mastery.

Use conceptual questions from several domains rather than arithmetic-only
skills. Initial families should include mechanics, electricity, heat and
matter, earth science, biology, and causal reasoning. Example states include:

- force is required to sustain motion;
- electric current is consumed by circuit components;
- heat is a material substance;
- seasons are caused by Earth–Sun distance;
- plant mass primarily comes from soil; and
- correlation is sufficient evidence of causation.

Within each family, include correct, uncertain, and at least one specific
misconception state. Keep the total state space small enough that every
transition can be inspected. Do not train one global classifier whose labels
are confounded with topic; evaluate state inference within each concept family
and macro-average across families.

### Student behavior

Use a deterministic state machine for reasoning and several templates for
verbalization. The environment, not a language model, owns:

- the student's current state;
- the error produced by each misconception;
- the effect of each tutor action;
- whether mastery changed; and
- delayed post-test performance.

Later experiments may replace the templates with a frozen language-model
verbalizer, but that model must not control state transitions.

### Tutor action space

The first experiment trains a structured pedagogical policy rather than
free-form tutoring. Candidate actions are:

- ask a diagnostic question;
- give a targeted hint;
- give a generic hint;
- demonstrate one step;
- reveal the answer.

The environment executes a canonical intervention for the selected action. This
prevents the tutor from declaring `targeted_hint` while emitting unrelated text.

### Gate 1: environment validity

Before collecting model rollouts:

- changing one hidden misconception while holding the problem fixed must change
  the student's error in the intended way;
- the same action must have different effects where the state says it should;
- revealing the answer may improve immediate correctness but must not count as
  delayed transfer;
- every state and action pair must have a tested transition; and
- replaying an episode with the same exogenous randomness must reproduce it.

Do not proceed if the environment cannot pass these intervention tests.

### Gate 1.5: state information must change the decision

Before collecting a large probe dataset, compare intervention selection from:

1. the full transcript;
2. the same transcript plus the gold misconception state; and
3. the gold state with a minimal problem summary.

Use the same examples and structured action set in every condition. Score
expert-action agreement and delayed transfer. If adding the gold state does not
improve either metric, explicit student-state inference is redundant for this
environment. Stop or redefine the state.

Represent the oracle state as an abstract conceptual relation, not the text of
a wrong answer option. The first 180-example pilot injected raw wrong-answer
text and reduced the gold-versus-believed margin by 0.30; inspection showed
that it anchored tutor generations on the wrong option's vocabulary.

## Phase 2: Test whether the model represents student state

Freeze the tutor model. The primary dataset should contain naturally occurring
errors: ask a student model to answer established concept-inventory questions
without naming a misconception, then assign labels from misconception-linked
distractors. Require a short explanation, and manually audit that the
explanation supports the selected label.

Do not probe the student generator's activations. Its causal states can directly
encode any system prompt used to control its behavior. Instead:

1. discard the student generation prompt and activations;
2. remove the choice letter and explicit final answer from the student response;
3. build a sanitized transcript containing only the problem and the remaining
   explanation;
4. run a separate frozen tutor/observer forward pass from scratch; and
5. extract observer activations after it reads the student response.

This removes direct prompt leakage. It does not remove lexical shortcuts in the
student response, so cross-prompt evaluation and surface controls remain
necessary.

Reuse the methodology from `projects/pedagogy_rm`:

- sweep middle layers;
- compare the final student-token state, the assistant-start state immediately
  before tutor generation, and mean pooling over the student explanation;
- fit a linear probe before trying an MLP;
- group folds by underlying problem; and
- score against simple surface baselines.

Surface baselines should include:

- student-answer correctness;
- bag-of-words features;
- response length;
- digit and symbol features; and
- the previous tutor action.

Hold out:

- question instances and templates;
- at least one topic per sufficiently broad misconception family;
- paraphrases of each misconception; and
- one complete student verbalizer or generator prompt.

### Gate 2: state decodability

Proceed only if:

- cross-template balanced accuracy materially exceeds the strongest surface
  baseline;
- the probe transfers to the held-out verbalizer;
- shuffled state labels return to chance; and
- the result is stable across at least three seeds.

A useful initial threshold is at least 10–15 accuracy points above the strongest
surface baseline. If this fails, either the state is not represented, the state
definition is poor, or the model is too small for the task.

Do not add an explicit belief prompt or auxiliary state loss during this phase;
that would test whether state can be implanted, not whether the native tutor
already represents it.

## Phase 3: Establish that state information has decision value

This phase expands Gate 1.5 into a policy-level test. Before RL, compare four
policies:

1. a history-blind action policy;
2. a policy conditioned on the dialogue;
3. an oracle policy given the true student state; and
4. a simple hand-authored policy.

All policies receive the same rollout and action budget. Evaluate them on a
fresh isomorphic problem after the tutor is removed.

### Gate 3: oracle-state gap

The oracle-state policy should outperform the ordinary dialogue-conditioned
policy by a practically meaningful margin. Use roughly 10 percentage points of
delayed transfer as the initial threshold.

If no oracle gap exists, better state inference cannot materially improve the
tutor. Stop or redesign the environment.

## Phase 4: Train belief-aware GRPO

Require structured policy output:

```text
<belief>force_implies_motion</belief>
<action>targeted_hint</action>
```

The environment verifies both fields exactly. The belief is the policy's
prediction, not privileged input.

Train with:

- terminal delayed-transfer reward;
- an answer-leakage penalty;
- deterministic belief-state correctness;
- a penalty for invalid actions; and
- group rollouts sharing the same problem and initial student state.

The belief prediction should occur before the action so that the model can use
it when selecting the intervention. The public natural-language response is
deferred until the structured policy works.

### Baselines

Compare against:

- terminal-transfer GRPO without belief prediction;
- belief tokens with shuffled state labels;
- belief supervision without RL;
- standard supervised action prediction;
- the hand-authored policy; and
- the oracle-state upper bound.

Use matched rollout-token and optimizer-step budgets.

### Primary success criterion

Belief-aware GRPO should close at least 25% of the delayed-transfer gap between
ordinary GRPO and the oracle-state policy without increasing answer leakage.

Also report:

- belief-state accuracy;
- delayed transfer;
- immediate correctness;
- retention after a longer delay;
- near-transfer to a related skill;
- leak rate;
- action distribution by true state; and
- worst-state performance, not only the mean.

## Phase 5: Robustness

Evaluate the trained policy on:

- unseen problem templates;
- unseen misconception wording;
- a different student verbalizer;
- longer conversations;
- delayed and near-transfer questions;
- altered transition probabilities; and
- new combinations of previously seen skills.

The hidden-state probe remains an independent diagnostic. Test whether improved
behavior coincides with clearer state representation, but do not use that
correlation as evidence of learning by itself.

Only after these tests pass should the project introduce:

- a frozen language-model student;
- free-form tutor responses;
- more detailed cognitive profiles; or
- human-student evaluation.

## Open-Instruct implementation

The proposed project layout is:

- `state.py`: hidden states, errors, and exact transitions;
- `env.py`: `TextRLEnvironment` implementation;
- `plugin.py`: environment and reward registration;
- `build_data.py`: RLVR rows;
- `extract_state_hidden.py`: activation extraction;
- `fit_state_probe.py`: diagnostic state probe;
- `benchmark.py`: oracle gap, transfer, and robustness evaluation;
- `tests/test_env.py`: transition and leakage tests; and
- `scripts/train.sh`: GRPO launch script.

Reuse:

- `open_instruct/scored_rewards/partner_env.py` for multi-turn conventions;
- `projects/pedagogy_rm` for activation extraction and probe-validation
  methodology;
- `open_instruct/scored_rewards/anchor.py` for held-out evaluation; and
- the `--reward_plugins` mechanism for project-specific registration.

The initial environment can return one terminal scalar, so it does not require
turn-level advantage changes. Snapshot and branch support can be added later if
the basic belief-aware policy succeeds.

## Falsification order

Run the project as three cheap gates before an expensive RL experiment:

1. **Environment validity:** do the controlled states produce the intended
   interventions and outcomes?
2. **State decodability:** does the small tutor represent the hidden state beyond
   surface cues?
3. **Decision value:** does oracle state information improve delayed transfer?

Only then run belief-aware GRPO.

Stop the project if any gate fails. A negative result at one of these stages is
more useful than a positive reward curve from an environment whose state,
measurement, or decision relevance was never established.

## Current execution status

The teacher-first mastery-graph MVP has now reached the stop condition above.
The graph and leakage-free structured-action SFT passed the state-comprehension
gate, but correct-graph interventions did not beat concept-only, shuffled-state,
or canonical-feedback controls on sealed ESTELA transfer items. GRPO is
therefore deferred. Full metrics and implementation paths are recorded in
`progress.md`; future work requires validated action-outcome data or a stronger
independently evaluated intervention generator.
