# Progress

Last updated: 2026-08-06

## Current objective

Test whether a small tutor/observer model's hidden states encode a student's
misconception after reading a student response.

The first deliverable is a probe result with grouped held-out accuracy, surface
baselines, and a learning curve. No RL training starts until this test is
credible.

## Decisions made

- Focus on small-model student-state inference rather than the three original
  post-training bets.
- Treat the hidden-state probe as a diagnostic, not as the learning-outcome
  reward.
- Use `Qwen/Qwen2.5-1.5B-Instruct` as the initial student generator because it
  is already configured in `projects/pedagogy_rm/generate.py`.
- Probe a separate observer/tutor forward pass, not the student generator's
  activations.
- The observer receives only a sanitized question and emitted student response.
  It never receives the misconception label, generator system prompt, or
  generator hidden states.
- Prefer post-hoc labels on naturally occurring mistakes. Controlled
  misconception conditioning is retained only as a secondary stress test.
- Use multiple conceptual domains rather than making arithmetic the research
  target. The completed arithmetic runs are instrumentation checks only.
- Split evaluation by underlying problem and hold out generator prompts or
  verbalizers. Random response-level splits are not acceptable.

## Leakage issue identified

If a student model is explicitly prompted to enact misconception `M`, its
generated-token activations can directly retain that instruction through causal
attention. A probe on those activations may decode the prompt rather than infer
the misconception from student behavior.

The primary fix is architectural separation:

1. Generate a student response.
2. Discard the generation context and all student-model activations.
3. Construct a clean transcript containing only the problem and response.
4. Run a separate frozen observer model from scratch on that transcript.
5. Extract observer activations after the final student token.

This removes direct prompt leakage. It does not remove lexical shortcuts in the
generated response, so the experiment also needs surface baselines and
cross-generator or cross-prompt evaluation.

## What a hidden student state can add

The observer hidden state is a deterministic function of the question and
student response. It cannot contain new evidence that was absent from that
text. The useful hypothesis is narrower: the transformer may convert distributed
textual evidence into a compact representation that makes the misconception
easier for the tutor policy to use.

Therefore, probe accuracy alone is not the goal. The project must distinguish:

1. **Information:** can any sufficiently capable text classifier infer the
   misconception?
2. **Accessibility:** is it linearly or cheaply decodable from the tutor's
   native hidden state?
3. **Decision value:** does knowing the state change which intervention should
   be selected?
4. **Use:** does an explicit belief bottleneck improve intervention choice or
   delayed learning over the ordinary transcript-conditioned tutor?

No steering is used for the accessibility test. Steering is justified only if
gold-state access improves decisions but the native tutor fails to exploit the
same information from the transcript.

## Data strategy

### Primary: naturally occurring errors

Use misconception-linked conceptual questions from several domains:

- mechanics: force is required to sustain motion; heavier objects fall faster;
- electricity: current is consumed; a battery supplies fixed current;
- heat and matter: heat is a substance; temperature measures total heat;
- earth science: seasons are caused by Earth–Sun distance;
- biology: plant mass comes from soil; individuals evolve because they need to;
  and
- causal reasoning: correlation is sufficient evidence of causation.

Prefer established concept-inventory items whose distractors already map to
documented misconceptions. Ask the student to answer unaided and explain its
reasoning without naming a misconception. Assign the label from its naturally
selected distractor, then mask the answer choice and explicit final answer
before the observer forward pass. The probe must infer the belief from the
explanation, not decode a choice letter.

Evaluate within concept family and macro-average across families. Otherwise the
question topic itself can reveal the label. Manually audit a stratified sample
to verify that the explanation actually supports the distractor-derived label.

### Secondary: controlled errors without label words

Programmatically create an incorrect intermediate trace using a known
transformation, then ask the student model only to verbalize that work. The
prompt should contain the work itself, not the misconception name.

Use this set to test coverage and sensitivity, not as the sole evidence that
misconceptions are naturally represented.

### Final stress test

If controlled prompting is used, train on some misconception prompts and test
on:

- unseen prompt templates;
- a different student generator;
- unseen problem templates; and
- paraphrases with different vocabulary.

## Data volume

The target is based on usable examples after filtering:

- pilot: 50 examples per class;
- minimum credible probe: 100 examples per class;
- preferred result: 200 examples per class;
- held-out test: at least 50 problems per class, grouped by problem.

For four to eight misconceptions plus correct responses, the preferred dataset
is at least 100–200 usable explanations per state, distributed across domains
and held-out question templates. Because natural generations will be imbalanced
and some explanations will contradict their selected answer, expect several
thousand generation attempts.

Fit learning curves at 25, 50, 100, and 200 training examples per class. Stop
collecting when the held-out confidence interval and learning curve stabilize,
not merely when a round-number target is reached.

## Evaluation contract

Report:

- balanced accuracy;
- macro F1;
- per-class precision and recall;
- 95% problem-cluster bootstrap confidence intervals;
- strongest surface-baseline performance;
- hidden-state gain over the surface baseline;
- results by layer and pooling; and
- learning curves by examples per class.

Required baselines:

- majority class;
- answer correctness alone;
- bag-of-words or TF-IDF logistic regression;
- simple length, punctuation, and digit features;
- observer hidden states with shuffled labels; and
- hidden states from an early layer.

The headline result is the grouped held-out hidden-state gain over the strongest
surface baseline, not raw training accuracy.

## Reusable local assets

- `projects/pedagogy_rm/generate.py`: existing student model and dialogue
  generation conventions.
- `projects/pedagogy_rm/extract_hidden.py`: multi-layer extraction and pooling.
- `projects/pedagogy_rm/probe.py`: grouped folds and surface controls.
- `projects/pedagogy_rm/README.md`: prior evidence that middle OLMo layers
  outperform final layers for pedagogy properties.
- `grpo_tutor/data/state_tests/train_items.jsonl`: 307 screened items where the
  0.5B student initially fails.
- `grpo_tutor/src/build_student_states.py` and `src/gen_traces.py`: attach
  item-specific wrong choices as `believes` / `believes_idx`.
- `grpo_tutor/runs/gen_traces_v2.jsonl`: 1,228 existing conditioned dialogues.

The existing extraction code reads tutor-turn states. It must be adapted to
extract observer states after the student response in a fresh sanitized
context.

The `grpo_tutor` belief fields are not a cross-item misconception taxonomy:
they identify which distractor was chosen for a particular question. Its
dialogues are also generated with that belief injected into the student prompt.
They are therefore useful as a secondary controlled stress test, but not as
evidence that a probe detects naturally occurring misconceptions.

## Status

- [x] Identify the research question.
- [x] Review the existing generation, extraction, and probe pipeline.
- [x] Identify direct misconception-prompt leakage.
- [x] Specify observer separation and post-hoc labeling.
- [x] Inspect available arithmetic/problem data and local model availability.
- [x] Implement the natural-error generator and deterministic labeler.
- [x] Generate an initial multiple-choice pilot: 80 problems x 3
  unconditioned samples using the locally cached
  `Qwen/Qwen2.5-0.5B-Instruct`.
- [x] Audit initial label quality and class balance. Only 113/240 responses
  were deterministically classifiable; minority classes had 18 and 19
  examples. Some selected distractors conflicted with the written reasoning,
  so this is not yet a credible "natural misconception" dataset.
- [x] Extract fresh observer hidden states for the initial pilot.
- [x] Fit initial surface and hidden-state probes.
- [x] Generate the stricter free-response pilot: 100 problems x 3 samples with
  no visible distractor options.
- [x] Report preliminary accuracy and confidence intervals.
- [x] Test whether an oracle wrong-belief label improves intervention choice
  over the full transcript alone. It did not; raw belief text made outcomes
  worse.
- [ ] Collect enough matched conceptual errors for a representation learning
  curve only if the oracle-state test shows decision value.

## Initial pilot result

The arithmetic multiple-choice pilot was an instrumentation check and is not
the intended research domain. It was also a failed/insufficient probe, not
positive evidence:

- middle-layer probe accuracy: 31.0%;
- middle-layer balanced accuracy: 28.1% (95% problem-bootstrap interval
  20.6%–36.0%);
- middle-layer macro F1: 27.9%;
- strongest TF-IDF surface baseline balanced accuracy: 32.5%; and
- middle-layer gain over that baseline: -4.4 percentage points.

Chance balanced accuracy is 25%. The interval includes chance and the hidden
state probe underperforms the surface baseline.

## Free-response instrumentation result

The free-response pilot produced:

- 300 attempts;
- 148 correct responses;
- 21 deterministically identifiable sign errors;
- 131 errors outside the predeclared taxonomy or unparseable responses; and
- zero natural operation-error or off-by-one examples.

For the binary correct-versus-sign-error probe:

- middle-layer raw accuracy: 87.6%, equal to the majority baseline's raw
  accuracy because the classes are highly imbalanced;
- middle-layer balanced accuracy: 70.4% (95% problem-bootstrap interval
  59.4%–81.0%);
- sign-error recall: 47.6%;
- sign-error F1: 48.8%; and
- strongest simple surface baseline balanced accuracy: 75.6%.

This is not evidence for a hidden-state advantage. The hidden probe is 5.1
percentage points below the strongest surface baseline, and the minority class
has only 21 examples.

At the observed 7% sign-error yield, approximately 1,400 attempts would be
needed for 100 sign errors. Do not scale this arithmetic dataset. The next
dataset should instead use matched, misconception-linked conceptual questions
across mechanics, electricity, heat, earth science, biology, and causal
reasoning. Explicit answers must be masked in a reasoning-only observer
condition. Collecting more of the current math data would mainly strengthen a
surface shortcut and would not answer the broader research question.

## Decision-value result: raw belief text failed

The oracle-state ablation ran on 180 non-math examples: 96 science and 84
social-studies questions. It compared tutor messages generated under:

1. the question and student utterance alone;
2. the same context plus the student's actual wrong belief; and
3. the same context plus a different wrong belief from the same question.

A frozen 0.5B student scores each answer choice after reading only the tutor
message. Primary metrics are paired changes in gold-answer probability,
gold-versus-believed margin, and solve rate. The counterfactual condition tests
whether any oracle gain is specific to the actual belief rather than merely
caused by adding extra diagnostic text.

These openings come from the existing controlled simulator and therefore test
the decision value of a state label, not whether the state occurred naturally.
The code is in `projects/student_state_tutor/oracle_state_ablation.py`.

The actual-belief condition did not help:

- transcript-only solve rate: 19.4%;
- actual-belief solve rate: 15.0%;
- paired solve-rate change: -4.4 points (95% CI -10.6 to +1.7);
- paired gold-probability change: -2.0 points (95% CI -4.6 to +0.5); and
- paired gold-versus-believed margin change: -0.30 (95% CI -0.50 to -0.12).

The margin degradation is statistically clear. Results remain negative when
both outputs flagged for answer leakage are removed. Science is approximately
neutral after that filtering, while social studies remains worse.

Inspection shows a mechanism: injecting the raw wrong answer anchors the tutor
on its vocabulary. The tutor often discusses or negates that answer rather than
providing the missing concept, which also reinforces those tokens for the
student scorer. A different wrong answer was no worse than the actual one, so
the effect is not evidence of misconception-specific targeting.

This fails Gate 1.5 for **raw item-level belief text**. Do not proceed directly
to hidden-state extraction or activation steering. The state definition must
first change from a wrong answer string to a compact conceptual relation, such
as `confuses light speed with frequency`. A subsequent oracle test should pass
that abstract relation without quoting any answer option. If that also fails,
explicit student-state modeling should be dropped.

Artifacts are in `projects/student_state_tutor/runs/oracle_nonmath/`.

## Redesigned next experiment

Change both the state representation and the outcome:

1. Curate four to six established conceptual misconception families, each with
   multiple diagnostic and isomorphic transfer questions.
2. Let the student answer the diagnostic question unaided and explain why.
3. Map a naturally selected distractor to an abstract relation such as
   `treats_current_as_consumed`, never to the distractor text itself.
4. Audit that the explanation supports the relation; discard contradictions.
5. Generate tutor interventions under transcript-only, gold-relation, and
   within-family counterfactual-relation conditions.
6. Evaluate the student on a different transfer question that shares the
   concept but not answer wording.

Initial candidate families:

- force is required to sustain motion;
- current is consumed by circuit components;
- heat is a material substance;
- seasons result from Earth–Sun distance;
- plant mass primarily comes from soil; and
- correlation implies causation.

Require at least 20 diagnostic/transfer pairs per family for a pilot and
100 naturally labelled responses per state before fitting a hidden-state probe.
Proceed only if the abstract gold relation improves transfer over both
transcript-only and counterfactual relations. If it passes, probe the native
assistant-start and middle-layer representations without steering. If it fails,
drop explicit student-state modeling.

## Active graph-state falsification

A cheaper prerequisite is running before transfer-pair curation. For each of
the 180 existing non-math examples, a frozen diagnostician converts the contrast
between the wrong and correct choices into an abstract two-field graph:

```text
concept = electric current
student_belief_relation = current is consumed by circuit components
```

States that quote any answer option are rejected. The tutor receives either the
actual relation or a relation derived from another wrong choice from the same
item. Both are compared against the already-generated transcript-only response.

This same-item test is not sufficient evidence for student-state value, but it
is a fast falsification gate:

- if the abstract actual relation still fails, do not curate transfer pairs;
- if it beats transcript-only and the counterfactual relation, proceed to the
  stronger diagnostic-to-transfer experiment.

Implementation:
`projects/student_state_tutor/graph_state_ablation.py`.

### Graph-state result

Of 180 examples, 154 produced valid actual/counterfactual graph pairs without
verbatim answer-option overlap.

The actual graph improved over transcript-only:

- gold probability: +3.8 points (95% CI +1.5 to +6.2);
- gold-versus-believed margin: +0.20 (95% CI +0.07 to +0.34); and
- solve rate: +6.5 points (95% CI 0.0 to +13.6).

However, the actual graph did **not** reliably beat a graph derived from a
different wrong option on the same question:

- gold probability: +0.9 points (95% CI -1.4 to +3.3);
- margin: +0.05 (95% CI -0.08 to +0.19); and
- solve rate: +2.6 points (95% CI -3.2 to +8.4).

This supports a weaker hypothesis: an abstract concept scaffold helps the tutor,
but there is not yet evidence that the student-specific relation matters.

### Active concept-only control

A follow-up compares:

1. transcript-only;
2. concept name only;
3. concept plus the actual mistaken relation; and
4. a shuffled concept from another item.

If concept-only matches the full graph, the gain is task decomposition rather
than student-state modeling. Only a full actual graph advantage over concept-only
would justify the diagnostic-to-transfer curation step.

### Concept-only result

Concept-only matched the full actual graph:

- full graph minus concept-only gold probability: +1.1 points
  (95% CI -1.7 to +3.7);
- margin: +0.03 (95% CI -0.14 to +0.18); and
- solve rate: +0.6 points (95% CI -6.5 to +7.8).

Concept-only itself beat transcript-only by +2.7 gold-probability points and
+0.17 margin, but did not reliably beat a shuffled concept. The current
single-turn evidence therefore supports generic task decomposition, not a
student-specific misconception relation.

The static single-response hypothesis is now rejected as a productive main
direction. The revised student-state hypothesis is:

> A persistent student-state graph is useful as memory when evidence about a
> student's belief is distributed across turns or absent from the latest
> response.

### Active persistent-memory experiment

The next falsification uses unresolved non-math dialogues with two to four
student turns. It compares the next tutor intervention under:

1. full dialogue history;
2. latest student turn only;
3. latest turn plus concept memory; and
4. latest turn plus concept and mistaken-relation memory.

The frozen student is scored on the same accumulated prior tutor context plus
the new intervention in every condition. This isolates whether a compact state
can recover useful information lost when dialogue history is unavailable.

Proceed only if relation memory beats concept-only and latest-turn-only. A
result that merely matches full history would still establish compression
value; failure against concept-only would reject this graph definition.

### Persistent relation-memory result

The mistaken-relation memory failed on 150 unresolved non-math dialogues:

- relation graph minus concept-only gold probability: -1.0 point
  (95% CI -3.1 to +1.1);
- margin: -0.06 (95% CI -0.19 to +0.07); and
- solve rate: -0.7 points (95% CI -6.7 to +5.3).

It also did not reliably beat the latest student turn alone. Full raw history
was worse than latest-turn-only, consistent with irrelevant or misleading
earlier dialogue reducing intervention quality. Concept-only achieved the best
point estimates.

This rejects both tested forms of static misconception state:

1. raw wrong-answer text; and
2. an abstract mistaken relation attached to one concept.

Do not fit hidden-state misconception probes from these labels. They have not
shown causal decision value.

## Revised hypothesis: mastery graph, not misconception label

The remaining promising student-state hypothesis is temporal and
decision-oriented:

> A persistent per-concept graph of mastery, uncertainty, and observed evidence
> helps a tutor choose what prerequisite to teach next across questions and
> sessions.

This differs from restating the student's latest error. A mastery graph can
contain information no longer present in the current turn, can be updated after
new evidence, and can prevent repeatedly teaching mastered concepts.

The next gates are structured rather than free-form:

1. **Oracle action value:** on multi-concept science and causal-reasoning tasks,
   test whether a gold mastery graph improves selection of the missing
   prerequisite over transcript-only and shuffled graphs.
2. **State update:** given a diagnostic response, test whether the model updates
   only the supported concept node while preserving unrelated mastered nodes.
3. **Transfer:** evaluate the selected intervention on a different question
   requiring the same missing prerequisite.

Only if all three pass should the project train a hidden-state readout or use
the graph during RL.

## Mastery-graph asset audit

The repository contains LLM-authored `units`, `prerequisites`, and
`misconception` metadata for all 186 non-math training items and 84 non-math
evaluation items. A leak-resistant graph build found:

- 618 distinct concept nodes and 350 prerequisite edges;
- 268 connected components; largest component size 7;
- only 11 repeated training nodes;
- 17 exact within-training transfer pairs; and
- only 3 exact train-to-evaluation shared nodes/pairs.

This graph is too fragmented for a broad transfer claim without semantic
canonicalization. Sixteen items also have metadata containing the full gold
answer text and must be excluded from model-visible state. Artifacts are in
`projects/student_state_tutor/runs/mastery_graph/`.

Public-data review found no single non-math dataset combining longitudinal
student responses, named misconceptions, and isomorphic transfer. The most
useful complementary resources are:

- SPHERE: 497 students answering 218 physics concept-inventory items across
  seven domains;
- BSCS/AAAS: explicit misconception mappings but restricted bulk access;
- BEETLE/SciEntsBank: free learner explanations but generic correctness labels;
  and
- ESTELA: 666 open isomorphic physics questions but no learner responses.

## Active real-human state validation

SPHERE is now used for a prediction-first gate on real students. For each of 20
random item splits, half of each physics inventory updates a Beta-Bernoulli
mastery node and the other half is held out.

Models compare:

1. item difficulty plus global student ability;
2. those features plus the correct per-inventory mastery node; and
3. those features plus mastery over randomly permuted item groups.

If the real mastery state does not improve held-out correctness prediction over
both controls, the mastery-graph direction is rejected before any tutoring or
RL experiment.

### Real-human mastery result

The prediction gate passed on 497 students and 218 physics items across 20
random item splits:

- item difficulty + global ability AUC: 0.753;
- adding the correct per-inventory mastery node AUC: 0.833;
- adding a random-group mastery node AUC: 0.753;
- mastery-state AUC gain: +0.080, with all item-split results between +0.075
  and +0.084;
- accuracy gain: +6.0 points; and
- log-loss improvement: 0.087.

This is the first strong positive evidence in the project. A persistent,
domain-specific mastery state predicts unseen responses far better than global
ability, while an equally sized random grouping adds nothing.

The result establishes predictive value, not tutoring value or causality. The
active robustness test uses mastery measured by one instrument to predict a
different but related instrument:

- FCI ↔ FMCE and FCI ↔ RRMCS for mechanics; and
- TCE ↔ STPFASL for heat/thermodynamics.

It compares related-instrument mastery against global ability, unrelated-domain
mastery, and student-shuffled mastery. Passing this gate would show that the
state transfers beyond test-specific response patterns.

### Cross-instrument result: selective, not general

The broad transfer claim did not pass. Pooled related-instrument mastery added
only +0.006 AUC over global ability and +0.003 over an unrelated-domain state;
the intervals included zero.

The effect depended on the pair:

- FCI → RRMCS: +0.015 AUC over global and +0.013 over unrelated;
- RRMCS → FCI: +0.017 over global and +0.016 over unrelated;
- FCI ↔ FMCE: about +0.004 over global but no advantage over unrelated; and
- TCE ↔ STPFASL: slightly negative.

Therefore, an entire inventory is too coarse to be a transferable concept node.
The positive within-instrument result is retained, but it must not be described
as general cross-context transfer. Future transfer tests need fine-grained,
isomorphic skill mappings.

### Curriculum decision-value result

The same graph was tested as a structured curriculum policy. Half of each
domain was used to estimate seven mastery nodes; disjoint held-out items defined
the student's actual weakest domain.

- predicted-weakest held-out mastery: 0.183;
- random-domain held-out mastery: 0.346;
- predicted-strongest held-out mastery: 0.640;
- weakest-domain top-1 identification: 53.7% versus 14.3% chance;
- domain-rank correlation: 0.621 versus 0.030 for shuffled nodes; and
- mean regret from the oracle weakest-domain choice: 0.058.

This passes the first structured action-value gate: the state changes the
decision of what content to assign next, and the chosen content is genuinely
weaker on unseen responses.

### State-update sample efficiency

The graph remains action-usable with sparse observations:

- 1 item/domain: 30.6% weakest-node identification, rank correlation 0.347;
- 2 items/domain: 32.6%, correlation 0.438;
- 4 items/domain: 37.1%, correlation 0.520;
- 8 items/domain: 45.3%, correlation 0.591; and
- 12 items/domain: 48.4%, correlation 0.614.

Even one observation per node beats chance and selects a domain whose held-out
mastery is 0.10 below a random choice. More evidence improves monotonically.

## Current supported hypothesis

The evidence now supports a narrower claim:

> A persistent, outcome-updated mastery graph is a useful state for selecting
> what a student should practice next.

It does **not** support static misconception relations, hidden-state steering,
or broad cross-instrument transfer. The state should initially be an explicit
Beta-Bernoulli evidence ledger updated from correctness, not a learned hidden
probe. The next causal gate is whether state-directed content selection improves
performance on held-out isomorphic questions relative to random or
global-ability curricula.

## Teacher-first architecture decision

Literature review and repository analysis support a simpler LLM integration:

> Keep student-state estimation explicit and probabilistic; use the graph as a
> compact input and action bottleneck for the LLM teacher.

This avoids asking one model to infer, update, simulate, teach, and judge the
same latent state. It also fits the limited-data regime: Bayesian knowledge
tracing and logistic models remain strong baselines, while current LLM student
simulators are controllable but weakly validated against human longitudinal
behavior.

The graph is split into six versioned stores:

1. concept registry;
2. typed taxonomy, prerequisite, and transfer edges;
3. item-to-concept Q-matrix;
4. immutable evidence events;
5. sparse per-learner posterior beliefs; and
6. optional embeddings used only for cold-start priors and retrieval.

The MVP does not attempt a universal topic ontology. It uses 20–50 ready,
text-only ESTELA isomorphic banks, with one mastery leaf per bank and unit
folders as organizational parents. Facets such as diagrams, contexts, and
misconceptions remain evidence tags until transfer results justify splitting a
node. Prerequisite edges retrieve candidates but do not propagate mastery.

### Active teacher-value gate

The next experiment evaluates a frozen LLM teacher before any SFT or GRPO:

1. diagnostic responses update explicit skill posteriors;
2. the teacher receives latest-turn, full-history, concept-only, correct-graph,
   shuffled-graph, or node-permuted context;
3. deterministic weakest-node and information-gain schedulers provide non-LLM
   baselines;
4. the teacher must first emit `{target_node, action, item_id, stop}`;
5. a separate generation step realizes the intervention; and
6. a sealed item from the same bank measures isomorphic transfer.

Run first with the existing frozen `grpo_tutor` student as a synthetic
falsification only. Stop the LLM branch if the correct graph does not change
structured actions, fails to beat transcript-only, or only matches the best
simple scheduler. Human intervention data remains mandatory for learning
claims.

Detailed architecture:
`projects/posttraining_paradigms/mastery_graph_architecture.md`.

## Teacher-first MVP implementation and result

The teacher-first architecture was implemented in
`projects/student_state_tutor/`:

- `estela_banks.py` extracts answer-keyed, text-only ESTELA banks;
- `mastery_state.py` implements assistance-aware Beta-Bernoulli beliefs,
  immutable evidence events, sparse serialization, weakest-node selection, and
  information-gain selection;
- `teacher_action_ablation.py` evaluates structured actions under latest-turn,
  full-history, concept-only, correct-graph, and shuffled-graph conditions;
- `teacher_action_sft.py` LoRA-fine-tunes the structured action policy;
- `teacher_transfer_ablation.py` screens the frozen student, generates
  interventions, and scores sealed isomorphic items; and
- the corresponding tests and Slurm launchers reproduce each stage.

All new unit tests pass. The unrelated installed Opik pytest shutdown hook emits
a `TypeError` after the test summary, but pytest exits successfully and the
tests themselves pass.

### ESTELA extraction audit

The checked-out data does support the planned catalog scale: canonical-file,
status, image, and answer-key validation found 21 ready, text-only,
auto-gradable banks with 455 questions. The normalized catalog includes 252
numerical, 70 multiple-choice, 51 multiple-answer, and 82 categorization
questions.

The current frozen-student evaluator scores one choice at a time and therefore
supports only the single-answer multiple-choice subset. Four banks and 70
questions passed that additional experiment-specific constraint:

- elevator Newton's second-law concepts: 15;
- static friction: 15;
- determining acceleration: 20; and
- simple harmonic motion expressions: 20.

Image-dependent, malformed, archival, and unfinished banks were excluded.
Numeric-response, multi-answer, and categorization items remain in the complete
catalog but were not silently converted into multiple-choice questions. The
four-bank transfer result is therefore a limitation of the current scorer, not
of ESTELA's available concept coverage.

### Frozen structured-action gate

On 32 held-out mastery profiles, the frozen Qwen2.5-3B teacher:

- selected the correct weakest node on 100% of valid correct-graph outputs;
- followed the visible weakest node in shuffled graphs 93.3% of the time;
- changed its target under graph permutation on 100% of applicable pairs;
- produced valid structured outputs on 84.4% of correct-graph cases; but
- selected the correct pedagogical action only 22.2% of the time.

The model understood and used the state, but made a systematic action-mapping
error. This met the architecture's condition for SFT, but not for GRPO.

### Structured-action SFT

The first SFT run was discarded after an audit found that synthetic learner IDs
contained the action label. That leakage produced low training loss without
reliable action generalization.

The corrected run used random action-independent learner IDs, multiple count
configurations around every policy threshold, 480 balanced examples, a
29.9-million-parameter LoRA adapter, 60 optimizer steps, and two epochs. On the
same held-out 32-profile gate it achieved:

- valid structured output: 100%;
- weakest-node accuracy: 100%;
- action accuracy given the correct target: 93.8%;
- shuffled visible-node following: 93.3%; and
- correct-versus-shuffled target sensitivity: 95.8%.

The structured-action gate therefore passed after leakage-free SFT.

### Synthetic isomorphic-transfer gate

The frozen Qwen2.5-0.5B student solved 15 of 70 ESTELA items during screening,
and every bank supplied at least two baseline errors. Thirty-two diagnostic /
sealed-transfer pairs were evaluated with no detected exact-answer leakage.

With the action adapter also used as the language generator, correct-graph
teaching did not improve transfer:

- graph gold probability: 0.1945;
- latest-turn: 0.1962;
- shuffled graph: 0.1955; and
- canonical ESTELA feedback: 0.2043.

Because the architecture separates action selection from language realization,
the experiment was repeated using the successful SFT action decisions but the
unmodified base teacher as the intervention generator:

- baseline transfer solve rate: 3.1%;
- graph solve rate: 15.6%;
- latest-turn solve rate: 12.5%;
- concept-only and shuffled-graph solve rates: 15.6%;
- canonical-feedback solve rate: 18.8%;
- graph minus latest gold probability: +0.0031,
  95% bootstrap interval [-0.0010, 0.0074]; and
- graph minus canonical gold probability: -0.0105,
  95% interval [-0.0190, -0.0025].

The transfer gate failed. The graph made action selection reliable, but the
resulting interventions were not state-specifically better than concept-only
or shuffled controls and were significantly weaker than the simple canonical
feedback baseline.

### Stop decision

GRPO was not run. This is the required outcome of the pre-registered
falsification order, not an incomplete training run: downstream RL would
optimize a synthetic reward after the student-state intervention had failed to
beat the non-RL baseline.

The supported result is now narrower:

> A compact mastery graph can make structured curriculum decisions easy for a
> small teacher model, and those decisions can be learned efficiently with
> supervised fine-tuning. The current evidence does not show that the graph
> improves free-form tutoring or isomorphic transfer.

A future continuation needs either real randomized intervention data or a
stronger independently validated action-conditioned generator. It should not
resume GRPO on the present synthetic student.

## Teacher-MoE implementation plans

Two implementation tracks now specify how to convert the teacher without
pretraining a new base model:

1. `dense_to_moe_upcycling_plan.md` describes a true FFN-MoE conversion with
   function-preserving expert cloning, sparse routing, continued pretraining,
   dense controls, and strict stop/go gates.
2. `teacher_lora_moe_plan.md` describes the recommended lower-cost path: four
   pedagogical-action LoRAs over one frozen Qwen teacher, hard routing per tutor
   turn, action-specific intervention data, and an oracle-action falsification
   before any learned router.

The teacher LoRA mixture should be implemented first. True MoE upcycling
proceeds only if oracle-routed action adapters show that specialization itself
adds value over one joint or parameter-matched wide adapter.

## Partial teacher MoE is operational

At the user's request, a small true-MoE infrastructure smoke was run before the
pedagogical-specialization gate. The implementation is in
`projects/dense_to_moe_upcycling/`.

### Data

A reproducible sampler streamed the community raw-text ClimbMix release rather
than downloading its roughly 400B-token corpus. The sample contains:

- 1,006,084 Qwen 2.5 tokens;
- 1,732 documents;
- eight semantic clusters, with roughly 200–250 documents per cluster; and
- source and SHA-256 metadata for every document.

The release is CC BY-NC 4.0. It is appropriate for this research smoke, not a
commercial training artifact.

### Architecture and run

The last two Qwen 2.5 3B teacher MLPs, layers 34 and 35, were replaced with four
cloned SwiGLU experts and a top-2 learned router. The rest of the teacher was
frozen. The resulting model has 3.492B resident parameters and 541.1M trainable
router/expert parameters.

Unit tests on a tiny Qwen model verified dense-function equivalence, router
gradients, expert dispatch, and checkpoint round trips. The H100 run then
trained for 50 optimizer steps, or 51,200 tokens.

Initial bfloat16 comparison:

- dense loss: 2.67008;
- upcycled loss: 2.66952;
- mean absolute logit difference: 0.00536; and
- maximum logit difference: 0.125.

Smoke-training result:

- initial held-out loss: 2.74258;
- final held-out loss: 2.74236;
- layer 34 load: 24.6%, 26.8%, 25.3%, 23.4%;
- layer 35 load: 28.8%, 22.2%, 24.1%, 25.0%; and
- dead experts: zero.

The checkpoint was restored over the dense base and generated a coherent
tutoring response, so conversion, sparse forward/backward, saving, restoration,
and inference are all operational.

### Important training fix

The first router run used gradient checkpointing while retrieving auxiliary
losses from module attributes. Those tensors were not connected to the active
checkpoint recomputation graph, so load balancing did not actually train the
router and one expert fell below 5% load. The corrected run disabled
checkpointing, logged nonzero router gradient norms, and balanced every expert
above 22%.

### Interpretation

This is a successful implementation smoke, not evidence that MoE improves the
teacher. The held-out loss movement is effectively zero at 51K tokens and no
dense continuation control was run. The next defensible comparison is partial
MoE versus matched dense continuation over at least 10M tokens, followed by
retention, throughput, expert-divergence, domain-routing, and downstream
teacher-action tests.
