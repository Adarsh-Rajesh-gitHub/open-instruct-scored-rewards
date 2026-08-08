# Routed teacher-expert LoRA mixture implementation plan

## Decision and claim

This is the recommended teacher-MoE implementation.

Keep one frozen `Qwen/Qwen2.5-3B-Instruct` teacher backbone and train four
pedagogical-action LoRAs. The explicit student-state policy selects one expert
for each tutor turn:

1. diagnostic question;
2. conceptual contrast;
3. worked example; and
4. retrieval practice.

`stop` remains a structured policy decision and does not need a language
expert.

This is a mixture of adapters rather than a sparse replacement of the dense
FFNs. The intended hypothesis is:

> Separating pedagogical actions into routed LoRA experts produces better
> state-appropriate interventions than one dense prompted teacher or one joint
> LoRA trained on all actions.

The current experiments make this the relevant bottleneck. The mastery graph
and structured action policy reached 100% target accuracy and 93.8% action
accuracy, but the generated interventions did not beat concept-only, shuffled,
or canonical-feedback controls.

## Why experts should represent actions, not subjects

There are only 21 ESTELA banks and 455 questions. Subject experts would divide
the data too aggressively and could route by surface vocabulary.

Pedagogical actions instead require genuinely different output structures:

- a diagnostic expert asks for evidence without teaching the answer;
- a contrast expert juxtaposes two cases and highlights a discriminating
  principle;
- a worked-example expert demonstrates a parallel problem step by step; and
- a retrieval expert asks the learner to reconstruct a principle or procedure.

Physics content and target concept remain in the prompt. The expert controls
the teaching operation.

## Architecture

```text
student responses
      |
external grader -> mastery graph
      |
structured policy -> {target_node, action, item_id, stop}
      |
hard action router
      |
Qwen2.5-3B base + selected action LoRA
      |
tutor intervention
      |
held-out model-student response
```

Routing is top-1 and fixed for one tutor turn. The structured policy already
knows the selected action; learning another router would add avoidable error.

The teacher never receives answer keys or the held-out transfer item.

## Repository layout

Create in `grpo_tutor`:

```text
src/
  teacher_experts.py
  build_teacher_expert_data.py
  train_teacher_expert.py
  evaluate_teacher_experts.py
  teacher_expert_router.py
tests/
  test_teacher_expert_router.py
  test_teacher_expert_prompts.py
  test_teacher_expert_scoring.py
scripts/
  build_teacher_expert_data.sbatch
  train_teacher_experts.sbatch
  evaluate_teacher_experts.sbatch
data/
  teacher_experts/
checkpoints/
  teacher-experts/
```

Reuse from `open-instruct/projects/student_state_tutor/`:

- `estela_banks.py --mode catalog`;
- `mastery_state.py`;
- `teacher_action_ablation.py`; and
- `teacher_transfer_ablation.py`.

Do not integrate the expert mixture into GRPO until the frozen/SFT action
experts pass their causal transfer gate.

## Phase 1: action-target dataset

### Source records

Use the full 21-bank, 455-question text-only ESTELA catalog. Each source record
includes:

- concept and unit;
- question;
- structured answer key;
- canonical feedback/solution when available;
- an isomorphic-bank identifier; and
- train/validation/sealed-template split.

The canonical solution is source material, not automatically a safe tutor
response. It often contains the answer and must not be shown verbatim for the
source assessment item.

### Four matched targets per context

For every training context, create four interventions while holding the
student evidence and target concept fixed:

#### Diagnostic

- one concise question;
- asks the student to expose a rule, comparison, or intermediate step;
- contains no explanation of the answer.

#### Conceptual contrast

- introduces two parallel cases;
- identifies the dimension that differs;
- asks the student to predict how the outcome changes; and
- avoids quoting the source choices.

#### Worked example

- uses a different numerical value, entity, or surface context;
- demonstrates the same principle;
- never solves the source or sealed transfer item; and
- ends with a connection question rather than the source answer.

#### Retrieval practice

- asks for the governing law, relationship, or procedure;
- is short enough to require active recall; and
- does not embed the requested response in the prompt.

Generate candidates with a stronger frozen model, then filter with deterministic
answer-overlap checks and independent rubric scoring. Keep source provenance
and generator version.

### Data splits

Split before target generation:

- training banks/templates;
- validation templates within known units; and
- sealed isomorphic templates or banks.

All four action versions of one context remain in the same split. Deduplicate
against sealed questions and answer keys.

Target approximately 1,200–2,000 accepted examples per action through safe
isomorphic variation. If fewer than 500 high-quality examples survive for an
action, do not train that expert yet.

## Phase 2: adapter training

### Initial configuration

Use the existing teacher LoRA stack as the baseline:

- base: `Qwen/Qwen2.5-3B-Instruct`;
- one independent adapter per action;
- LoRA rank: 16 initially;
- alpha: 32;
- dropout: 0.05;
- target modules: all linear projections;
- assistant-token-only cross-entropy;
- maximum sequence length: 1,024;
- batch size: 4 with gradient accumulation to effective 16;
- learning rate: compare `5e-5` and `1e-4`;
- 2–3 epochs; and
- selection by held-out intervention metrics, not training loss.

Every expert starts from the same frozen base. Do not continue one action
adapter from another.

### Shared preservation replay

Mix 20% general tutoring responses into each expert's training data. Distill
the base teacher on unrelated prompts if an expert develops narrow formatting
or loses general instruction following.

### Checkpoint manifest

Store:

```text
base_model
base_revision
adapter_name
action
lora_rank
lora_alpha
target_modules
dataset_version
prompt_version
training_seed
```

All resident adapters must have compatible target modules and rank.

## Phase 3: hard expert routing

Implement `TeacherExpertRouter`:

- validates the structured action;
- maps action to adapter name;
- selects that adapter before generation;
- groups a batch by adapter;
- generates homogeneous sub-batches;
- restores original order; and
- exposes the no-adapter base route.

For initial evaluation use Transformers/PEFT generation. Once behavior is
validated, vLLM multi-LoRA can serve one adapter per request. Do not merge the
adapters into the base because that destroys routing.

The router should reject unknown actions rather than silently falling back to a
random expert.

## Phase 4: oracle-action falsification

Before using inferred mastery state, supply the correct pedagogical action and
compare:

1. dense base with an action instruction;
2. one joint rank-16 LoRA conditioned on action text;
3. one joint wide LoRA matched to the total expert parameters;
4. four LoRAs with the correct hard action route;
5. four LoRAs with a shuffled/wrong action route;
6. four duplicated adapters under different names; and
7. canonical ESTELA feedback.

Use identical examples, tokens, optimization steps, seeds, and search budgets.

### Generation metrics

- action adherence;
- answer and option leakage;
- source-answer overlap;
- intervention specificity;
- response length and cost;
- factual correctness;
- distinction between correct and shuffled experts; and
- worst-action performance.

### Causal metric

Score a frozen model student on a different item from the same bank after
receiving only the intervention. Report:

- solve rate;
- gold probability and margin;
- graph/action versus shuffled controls;
- improvement over dense base and joint LoRA;
- improvement over canonical feedback; and
- bootstrap intervals clustered by bank, not by repeated rollout.

The routed mixture must improve held-out responses, not merely make its action
easy to classify.

## Stop/go gates

An individual expert passes only if:

- action adherence is at least 90%;
- answer leakage is at most 5%;
- factual-error rate does not exceed the dense base;
- performance holds on sealed templates; and
- no expert collapses to a fixed generic script.

The mixture hypothesis passes only if correct hard routing:

- beats shuffled routing by at least 5 solve-rate points;
- beats the joint LoRA on mean gold probability in at least three seeds;
- beats or matches the parameter-matched wide LoRA;
- does not lose to canonical feedback by more than the predefined practical
  margin; and
- improves worst-action performance rather than only the mean.

If oracle hard routing cannot beat one joint LoRA, stop. A learned router or
true FFN MoE is unlikely to help this data regime.

## Phase 5: graph-conditioned evaluation

Only after oracle action routing passes:

1. update the mastery graph from diagnostic outcomes;
2. let the structured policy select the target and action;
3. route to the matching teacher expert;
4. generate the intervention; and
5. evaluate on sealed isomorphic items.

Conditions:

- latest response only;
- full history;
- concept only;
- correct graph;
- shuffled graph;
- deterministic weakest-node scheduler;
- deterministic information-gain scheduler; and
- oracle state/action.

This separates state estimation, action selection, and intervention
realization.

## Phase 6: optional learned routing

Learned routing is justified only if prompts do not already contain the
structured action or if composite interventions require multiple experts.

### Sequence-level router

Train a small classifier over the retrieved state summary to predict one action
expert. Compare directly with the existing structured-action SFT, which already
reached 93.8% action accuracy.

### Weighted mixture

For actions such as `contrast + retrieval`, combine adapter deltas:

```text
delta(x) = sum(alpha_i * B_i A_i x)
```

Validate mixtures behaviorally. Do not separately average `A` and `B`.

### Token-level routing

Do not use token-level routing unless a sequence-level mixture demonstrably
fails on compositional actions. Token routing makes pedagogical intent harder
to audit and can switch strategies mid-sentence.

## Phase 7: teacher post-training

If the expert mixture improves causal transfer:

1. freeze experts and train only the structured router with SFT;
2. compare a contextual bandit over discrete actions;
3. run GRPO only if policy errors remain after SFT; and
4. keep the intervention experts fixed during the first policy experiment.

Later, expert-specific GRPO can update only the selected adapter, but every
sample must be compared against:

- frozen expert SFT;
- one joint GRPO LoRA;
- no-graph GRPO; and
- the canonical-feedback baseline.

## Compute estimate

Four rank-16 LoRAs over the 3B base fit on one L40S or H100. Train experts
sequentially to keep optimizer memory low; load all four only for inference.

Expected after data generation:

- one action-expert pilot: roughly 1–2 L40S/H100 hours;
- four SFT experts and controls: roughly 6–12 GPU-hours;
- held-out causal evaluation: roughly 1–2 GPU-hours.

Data quality is the main risk. More experts will not fix interventions that are
factually wrong, answer-revealing, or ineffective for the frozen partner.
