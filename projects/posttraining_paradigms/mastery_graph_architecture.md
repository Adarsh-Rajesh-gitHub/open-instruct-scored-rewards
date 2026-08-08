# Teacher-first mastery graph architecture

## Decision

The most promising use of student state is not to make an LLM imitate a
student's cognition. It is to give an LLM teacher a compact, evidence-backed
view of what to teach next.

The proposed system has three separate responsibilities:

1. A probabilistic estimator updates student state from observed responses.
2. An LLM teacher reads a small relevant portion of that state and chooses a
   teaching action.
3. A real student, or a clearly labelled synthetic test fixture, produces the
   next response.

The estimator owns state. The LLM does not write its own mastery scores. A
synthetic student does not define whether teaching succeeded.

This separation follows the empirical results so far:

- Static wrong-answer and misconception-relation states did not improve
  tutoring.
- A simple per-domain mastery state strongly predicted held-out human
  responses and weak-domain curriculum choices.
- Broad cross-instrument transfer was mostly null, so the graph cannot assume
  that coarse topic labels are transferable cognitive units.

The resulting thesis is:

> A persistent mastery graph can act as an evidence and data-routing layer
> that helps an LLM teacher select and realize a better next intervention.

## Prediction, teaching, and simulation are different claims

### Predicting a real student

The estimator models:

```text
P(next real response | prior responses, item, assistance, time)
```

This is a supervised forecasting problem. It can be tested with held-out human
responses using log loss, Brier score, calibration, and AUC.

### Teaching a real student

The teacher policy models:

```text
P(action | estimated state, available content, time budget)
```

It must ultimately be tested by an intervention: did the chosen action improve
unaided performance on a different item after a delay?

### Simulating a student

A simulator attempts to model:

```text
P(response, next state | current state, tutor intervention)
```

This is much harder because it requires counterfactual learning dynamics.
Current LLM simulators can follow explicit ability or misconception controls,
but prompt compliance is not evidence that their learning transitions match
humans. A simulator is therefore a fast falsification environment, not the
scientific ground truth.

## System flow

```text
student response
    |
external grader and item rubric
    |
immutable evidence event
    |
Bayesian state updater <--- versioned item-to-skill map
    |
sparse learner mastery state
    |
retrieve weak/uncertain nodes, validated prerequisites, and candidate items
    |
LLM teacher emits {target node, action, item, stop}
    |
intervention generator realizes the action
    |
student receives intervention
    |
held-out isomorphic and delayed evaluation
```

The teacher sees only deployment-available evidence. It never sees held-out
questions, answer keys, future responses, or gold misconception labels.

## The graph is six versioned stores

Do not overload one graph with taxonomy, prerequisites, item mappings, and
student state.

### 1. Concept registry

```text
Concept {
  concept_id
  version
  preferred_label
  description
  kind: collection | assessable_skill
  action
  object
  conditions
  status: active | deprecated
}
```

Concept IDs are permanent. Renaming creates an alias; splitting and merging
create versioned migrations.

### 2. Typed concept edges

```text
ConceptEdge {
  source_id
  target_id
  type: broader | prerequisite | transfer | analogy
  confidence
  context
  provenance
  graph_version
}
```

These edge types are not interchangeable:

- `broader` organizes a taxonomy.
- `prerequisite` proposes that one capability supports learning another.
- `transfer` says evidence generalizes between capabilities.
- `analogy` helps retrieve pedagogically useful comparisons.

Initially, edges retrieve candidates and modify priors. They do not copy
correctness into neighboring mastery nodes.

### 3. Item-to-concept mapping

```text
ItemConcept {
  item_id
  item_version
  concept_id
  role: required | supportive | diagnostic
  loading
  confidence
  provenance
}
```

This is the Q-matrix. It is separate from the prerequisite graph. For the first
experiment, prefer single-skill items. One multi-skill response must not be
duplicated as independent evidence for every attached concept.

### 4. Immutable evidence ledger

```text
EvidenceEvent {
  learner_id
  item_id
  timestamp
  outcome
  assistance: unaided | hinted | worked | revealed
  latency
  evaluator_version
  rubric_version
  reliability
  graph_version
}
```

Unaided, hinted, and answer-revealed responses are different evidence channels.
An answer revealed by the tutor must not establish mastery.

### 5. Sparse learner beliefs

```text
LearnerConceptState {
  learner_id
  concept_id
  posterior_mean
  posterior_variance
  effective_evidence
  unaided_evidence
  transfer_evidence
  last_observed_at
  half_life
  state_version
}
```

Only encountered concepts are materialized. A million-concept registry does not
create a million entries for every learner.

### 6. Optional continuous priors

Concept and learner embeddings may initialize cold-start priors or retrieve
candidate neighbors. They are not authoritative mastery evidence. A semantic
similarity between two topics is not proof that a student transfers between
them.

## What a mastery node means

A node should not mean "recent percent correct." Define it operationally:

> The posterior probability that this learner can independently solve a novel,
> isomorphic item requiring the skill, at a reference difficulty and retention
> delay.

The state remains continuous. Labels such as these are derived views:

- `unassessed`: too little evidence;
- `needs_work`: credible interval below the criterion;
- `developing`: interval overlaps the criterion;
- `mastered`: lower credible bound exceeds the criterion.

Unknown is not the same as low mastery.

## How granular should nodes be?

Use three levels.

### Collections

Examples: `physics`, `mechanics`, `forces`.

Collections organize content and provide hierarchical priors. They are too
broad to carry primary mastery claims.

### Assessable skill leaves

A leaf is an action-object-condition capability, for example:

```text
identify Newton's third-law interaction pairs in text-only equilibrium scenarios
```

For the MVP, one curated isomorphic problem bank is one leaf. The bank's unit
folder is its parent collection. This lets available transfer evidence define
granularity instead of inventing a universal ontology in advance.

### Facets and evidence tags

Examples:

- diagram versus prose;
- incline versus horizontal surface;
- multiple choice versus explanation;
- language and reading load;
- a supported misconception relation.

Facets are not separate mastery nodes until data shows that they have distinct
transfer patterns and require different interventions.

### Split and merge rules

Split a node when all of the following are true:

1. Errors or transfer outcomes separate reliably by context or representation.
2. The proposed children require different remediation.
3. Each child has enough independent items and learner observations.
4. Held-out prediction improves after accounting for added complexity.

Merge nodes when they have indistinguishable transfer and learning curves, the
same intervention, and similar graph neighborhoods.

Lexical synonyms become aliases, not separate nodes.

When splitting a node, initialize children from the parent but increase
uncertainty. Historical evidence did not distinguish the children.

## State update

### Transparent baseline

Begin with a weighted Beta-Bernoulli posterior:

```text
alpha[k] <- alpha[k] + reliability * unaided_correct
beta[k]  <- beta[k]  + reliability * unaided_incorrect
mean[k]  = alpha[k] / (alpha[k] + beta[k])
```

This baseline already showed useful prediction and curriculum value in SPHERE.
It requires little data and makes every update auditable.

Use hierarchical priors so sparse leaves shrink toward their parent and
population means. Do not fit independent parameters for hundreds of rare
skills.

### Item-aware challenger

Correctness on an easy item is weaker evidence than correctness on a hard item.
The next challenger should combine dynamic IRT and Bayesian knowledge tracing:

```text
P(correct) =
  guess +
  (1 - guess - slip) *
  sigmoid(discrimination * (proficiency - difficulty))
```

The posterior is updated from the residual between observed and predicted
performance. This separates:

- learner proficiency;
- item difficulty;
- item discrimination;
- lucky guessing;
- careless slipping.

### Forgetting

Only add forgetting when timestamps and repeated delayed evidence can estimate
it. A simple mean-reverting model is:

```text
retention = exp(-elapsed_time / skill_half_life)
decayed_mean = prior_mean + retention * (old_mean - prior_mean)
```

Uncertainty should increase as evidence becomes stale.

### Prerequisites

Prerequisites can influence:

- which diagnostic to ask;
- which candidate item to retrieve;
- the prior probability of learning a child skill.

They should not automatically mark a parent mastered after a child success or
mark every prerequisite unmastered after a child failure.

## LLM teacher interface

The teacher should receive a bounded neighborhood, not the entire graph:

```json
{
  "current_goal": "newton_third_law",
  "nodes": [
    {
      "concept_id": "identify_interaction_pairs_text",
      "mean": 0.31,
      "interval": [0.16, 0.50],
      "unaided_n": 4,
      "last_seen_days": 3
    },
    {
      "concept_id": "distinguish_balanced_forces_from_action_reaction",
      "mean": 0.44,
      "interval": [0.24, 0.65],
      "unaided_n": 3,
      "last_seen_days": 1
    }
  ],
  "recent_evidence": [
    {
      "concept_id": "identify_interaction_pairs_text",
      "correct": false,
      "assistance": "unaided"
    }
  ],
  "candidate_items": ["n3_bank_q08", "n3_bank_q14"],
  "remaining_turns": 2
}
```

The first output is structured:

```json
{
  "target_node": "identify_interaction_pairs_text",
  "action": "contrast_case",
  "item_id": "n3_bank_q14",
  "stop": false
}
```

A separate generation step realizes that action as natural language.

This two-stage design determines whether the state improved selection or merely
changed wording. It also makes the LLM directly comparable to deterministic
schedulers.

## Why prioritize the teacher over the simulator?

The graph has already shown that it can select weak content. That information
can directly alter a teacher's action.

A state-conditioned simulator is easier to control but harder to validate.
Prompting a model with `mastery = low` can make it answer incorrectly by
construction. That proves compliance, not human realism. LLM student simulators
also tend to:

- know too much;
- accept generic correction too readily;
- overproduce articulate explanations;
- underproduce disengagement and persistent errors;
- become more coherent as additional context is added.

If a simulator is used, separate:

1. the external latent state and transition model;
2. the intent planner with a closed behavior set; and
3. the LLM response renderer.

Never let the renderer see answer keys or future outcomes. Validate the
simulator against human calibration, error persistence, learning curves,
forgetting, and intervention-effect rankings.

## First teacher experiment

### Data

Use 20–50 ready, text-only banks from the open
[ESTELA physics problem bank](https://github.com/Zhongzhou/ESTELA-physics-problem-bank).
Each bank supplies:

- diagnostic items;
- intervention/practice items;
- held-out isomorphic transfer items.

One bank is one initial leaf skill. Exclude banks with figures, ambiguous answer
keys, or items that are only numerical substitutions.

### Student histories

Administer diagnostic items to the existing frozen `grpo_tutor` student.
Update the explicit posterior from scored answers. Keep the same student
configuration and random seed across compared teacher conditions.

This is a synthetic fast falsification. It cannot support claims about human
learning.

### Teacher conditions

Compare:

1. latest response only;
2. full raw history;
3. concept name only;
4. correct mastery-graph neighborhood;
5. another student's graph;
6. node-permuted graph;
7. weakest-posterior deterministic scheduler;
8. information-gain deterministic scheduler; and
9. oracle state/action upper bound.

All conditions receive the same candidate items, generation budget, and
number of turns.

### Two evaluation stages

#### Stage A: action sensitivity

Before generating tutoring text, score:

- selected target node;
- selected action;
- selected item;
- regret relative to the oracle action;
- sensitivity to correct versus shuffled state;
- stability under paraphrased graph rendering.

The graph has no LLM value if the teacher ignores it or reacts identically to a
shuffled state.

#### Stage B: isomorphic transfer

Generate the intervention, then test the frozen student on a sealed item from
the same bank. Measure:

- unaided transfer solve rate;
- gold-answer probability and margin;
- answer leakage;
- intervention length and cost;
- worst-bank performance;
- robustness across student seeds and teacher samples.

Immediate repetition of the diagnostic item is not transfer.

### Stop/go gates

Stop the LLM branch if:

- correct and shuffled graph states produce the same actions;
- the graph-conditioned teacher fails to beat transcript-only;
- it only matches the best deterministic scheduler; or
- gains disappear on held-out banks or under graph paraphrases.

Proceed to supervised fine-tuning only if the frozen model understands the
state but makes a systematic, learnable selection error.

Proceed to GRPO only if:

- graph-conditioned SFT still leaves meaningful action regret;
- delayed/isomorphic rewards can be computed independently;
- graph-GRPO beats graph-SFT and no-graph GRPO under equal budgets; and
- results replicate across independently specified student fixtures.

Human intervention data remains mandatory before claiming learning gains.

## Data-limited strategy

Limited data argues for stronger separation and simpler estimators:

- Use existing isomorphic banks instead of generating an ontology first.
- Let bank membership define initial leaf nodes.
- Partially pool sparse leaves toward unit parents.
- Use LLMs to propose item mappings and variants, but verify mappings and
  answer keys independently.
- Prefer uncertainty-aware item selection to training a high-capacity policy.
- Treat every real response as a reusable evidence event.
- Collect randomized action-outcome data before fitting a simulator.
- Try contextual bandits or Thompson sampling before full RL.

Learning science enters primarily through the data policy and evaluation:

- spacing;
- interleaving;
- prerequisite remediation;
- desirable difficulty;
- value of information;
- assistance-aware evidence;
- isomorphic transfer;
- delayed retention.

It does not require modifying the Transformer architecture.

## Repository integration

### Existing student fixture

`/Users/sophiaz/_RL/grpo_tutor/src/student_state.py` currently stores a
per-problem three-level `mastery`. Keep it for simulator behavior, but add the
persistent graph as a separate object. Do not silently reinterpret that field
as longitudinal human mastery.

### Environment

Implement the future graph-backed director against:

`/Users/sophiaz/_RL/open-instruct/open_instruct/scored_rewards/partner_env.py`

Its `observe()` method should append scored evidence. A separate updater
changes learner beliefs. The director can use those beliefs to control the
synthetic student's intent, but the tutor receives only its allowed graph view.

### Existing human baselines

Reuse:

- `sphere_mastery_prediction.py`;
- `sphere_curriculum_value.py`;
- `sphere_state_update_curve.py`; and
- `sphere_cross_instrument.py`.

These establish predictive and curriculum value, and document the failure of
coarse cross-instrument transfer.

### GRPO integration

Only after the teacher gates pass:

- expose the structured action through an open-instruct project environment;
- fork every GRPO sample from the same serialized graph snapshot;
- reward sealed isomorphic or delayed transfer;
- penalize answer leakage and unnecessary intervention cost; and
- compare graph-GRPO against graph-SFT, no-graph GRPO, and simple schedulers.

## MVP execution outcome

The architecture was implemented and run through its synthetic gates.

The available ESTELA checkout supports 21 ready, text-only, auto-gradable banks
with 455 questions. The current frozen-student scorer handles only
single-answer multiple-choice outcomes, reducing the executed transfer pilot to
four banks and 70 questions across elevator forces, static friction,
acceleration, and simple harmonic motion. The pilot limitation is in the
evaluation interface rather than the source catalog.

The frozen teacher read the graph correctly: it selected the weakest node with
100% accuracy and changed its target under shuffled states. It did not map
posterior patterns to the required action reliably. A leakage-free,
480-example LoRA SFT fixed this narrow structured task, reaching 100% valid
output, 100% target accuracy, and 93.8% action accuracy on 32 held-out states.

The downstream gate did not pass. On 32 diagnostic / held-out isomorphic pairs,
the graph-conditioned policy with a separate base intervention generator
reached 0.1941 mean gold probability and 15.6% solve rate. It did not separate
from concept-only or shuffled-state controls. Canonical ESTELA feedback reached
0.2046 and 18.8%; graph minus canonical was -0.0105 with a 95% bootstrap
interval of [-0.0190, -0.0025].

Therefore GRPO is intentionally blocked. The implemented graph has demonstrated
state compression and structured decision usability, not causal tutoring
value. Resuming policy optimization requires a validated action-conditioned
generator or real randomized action-outcome data.

## Research grounding

Relevant foundations and cautions include:

- [Knowledge Tracing](https://doi.org/10.1007/BF01099821): interpretable
  Bayesian mastery updates.
- [Performance Factors Analysis](https://eric.ed.gov/?id=ED506305):
  scalable success/failure-count baselines.
- [Dynamic Bayesian Networks for Student Modeling](https://cgl.ethz.ch/Downloads/Publications/Papers/2017/Kae17a/Kae17a.pdf):
  dependent skill state with richer transitions.
- [Deep Knowledge Tracing](https://proceedings.neurips.cc/paper_files/paper/2015/file/bac9162b47c56fc8a4d2a519803d51b3-Paper.pdf):
  flexible sequence models without calibrated symbolic state.
- [AKT](https://arxiv.org/abs/2007.12324): attention-based knowledge tracing
  with monotonic decay and item variation.
- [PSI-KT](https://proceedings.iclr.cc/paper_files/paper/2024/file/99238c9d7ad8c6d138dc417fd8e3740c-Paper-Conference.pdf):
  hierarchical generative knowledge tracing and inferred prerequisites.
- [PEARL](https://arxiv.org/abs/2605.29582): structured student simulation and
  pedagogically aligned RL, with limited human simulator validation.
- [Towards Valid Student Simulation](https://arxiv.org/abs/2601.05473):
  explicit epistemic state specifications and validation requirements.
- [SPHERE](https://doi.org/10.1038/s41597-025-04913-0): the real-human physics
  response dataset used for the current mastery-state evidence.

## Bottom line

The graph should be simple where data is scarce and explicit where decisions
matter. The immediate question is not whether a large model can internally
simulate cognition. It is:

> Does a correct, compact student-state summary cause an LLM teacher to choose
> and deliver a better intervention than raw history or a simple scheduler?

That question is falsifiable with the current codebase and available isomorphic
item banks. It is the highest-probability path to a useful result.
