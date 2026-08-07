"""ORM and PRM as rewards open-instruct GRPO can call.

    python open_instruct/grpo_fast.py \
        --reward_plugins projects/prm_vs_orm/plugin.py \
        --group_scorer orm:head=<ckpt>      # outcome reward
        # or
        --group_scorer prm:head=<ckpt>      # process reward

This is the one experiment-defining seam of the PRM-vs-ORM study: the same GRPO
loop, the same policy, driven once by an *outcome* reward model and once by a
*process* reward model, both trained from PRM800K. Everything else about the run
is held fixed so the reward is the only thing that differs.

THE TWO REWARDS, AND HOW A NUMBER COMES OUT OF EACH.

``orm`` (outcome). A classifier fine-tuned on whole solutions reads the LAST
response token and emits one logit; ``sigmoid`` turns it into ``P(solution
correct)``. That single probability is the reward for the whole completion --
Cobbe et al. 2021's verifier, read at the end because a causal model has only
then seen the whole solution. Nothing about the interior of the solution enters
the score.

``prm`` (process). A classifier fine-tuned on individual reasoning steps reads
the last token of each step and emits three logits -- negative / neutral /
positive (Lightman et al. 2023's PRM800K labels). ``P(step ok) = P(pos) +
P(neu)`` (neutral counts as positive, per the paper). The completion's reward is
the PRODUCT of the per-step probabilities: one bad step should sink the whole
solution, which a mean would not do. ``min`` and ``mean`` are computed too and
returned in ``info`` so the aggregation choice can be revisited without a rerun;
the product is reported to be slightly biased against longer solutions, which is
why the alternatives are kept in view.

THE BACKBONE IS FROZEN AND SEPARATE FROM THE POLICY. Both reward models are their
own copy of a model, loaded once inside the scoring actor and never updated. A
reward model that trained alongside the policy would be a moving target, and a
head reading the policy's own trunk would hand the policy a text-free channel to
raise its reward -- the failure the pedagogy project documents at length in
``projects/pedagogy_rm/plugin.py``.

WHAT IS AND IS NOT IMPLEMENTED HERE YET. The reward *arithmetic* above is real
and unit-tested (see ``selftest.py``). The *backend* that turns text into model
logits is deferred: it loads a checkpoint whose format is defined by the reward
trainer in Stage 3-4 (``reward_modeling_scored.py``), which does not exist yet,
so ``head=<path>`` raises with that pointer rather than guessing the schema. A
torch-free ``stub=true`` backend stands in for the plumbing smoke -- it produces
finite, in-range, MEANINGLESS numbers whose only job is to prove the plugin
imports, registers and scores on the platform image. It must never drive a real
run; ``head=`` is the only backend that carries signal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from open_instruct.scored_rewards import Sample, Scorer, ScoreResult, register

#: How a completion is split into reasoning steps. PRM800K writes one step per
#: paragraph, so the boundary is a blank line, and the final ``# Answer`` block is
#: its own (last) step. data_prep.py MUST emit steps against this same separator or
#: the PRM's step-boundary token positions will not line up with what it scored at
#: training time -- the single most important cross-stage invariant of the PRM arm.
STEP_SEP = "\n\n"


def _solution_view(sample: Sample) -> str:
    """The exact string a reward model reads: the problem, then the solution.

    Both reward models are trained on ``prompt + completion`` (a solution is only
    correct or incorrect relative to its question), so scoring must reconstruct
    the same concatenation. Keeping this in one place means the ORM and the PRM,
    and training and inference, cannot drift apart in how they frame an input.
    """
    prompt = sample.prompt or ""
    completion = sample.completion or ""
    return f"{prompt}{completion}" if prompt.endswith(("\n", " ")) or not prompt else f"{prompt}\n\n{completion}"


def _steps(completion: str) -> list[str]:
    """Split a completion into the steps the PRM scores, dropping empties."""
    return [s for s in (completion or "").split(STEP_SEP) if s.strip()]


# --------------------------------------------------------------------------------
# Backends. A backend maps text -> the model's raw quantity. The real one loads a
# trained reward model; the stub is torch-free and carries no signal.
# --------------------------------------------------------------------------------


def _stub_prob(text: str) -> float:
    """A deterministic, in-``(0, 1)``, MEANINGLESS probability. Plumbing only.

    Derived from the text length alone so a smoke run is reproducible. It exists
    so ``score_group`` can be exercised end to end without a checkpoint or a GPU;
    it says nothing about whether the text is any good, and a run that optimises
    it optimises noise.
    """
    n = len((text or "").split())
    return 1.0 / (1.0 + math.exp(-(0.05 * (n % 20) - 0.5)))


def _deferred_backend(head: str) -> Callable[..., object]:
    """The real reward-model backend, which Stage 3-4 will supply.

    Raising here, rather than shipping a plausible-looking loader, keeps the
    checkpoint schema a decision made once in ``reward_modeling_scored.py`` and
    read here, instead of guessed in two places that then disagree.
    """

    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            f"the trained reward-model backend is not implemented yet (asked for head={head!r}). "
            "It loads the checkpoint written by projects/prm_vs_orm/reward_modeling_scored.py "
            "(Stage 3-4). Until then, pass stub=true for the plumbing smoke."
        )

    return _raise


class OrmScorer(Scorer):
    """Outcome reward: one ``P(solution correct)`` per completion, read last-token."""

    name = "orm"

    def __init__(self, head: str = "", stub: bool = False, device: str = "cuda", max_length: int = 2048):
        self.head_path = head
        self.stub = bool(stub)
        self.device, self.max_length = device, int(max_length)
        if self.stub:
            #: text -> P(correct) for a whole solution.
            self._prob: Callable[[Sequence[str]], list[float]] = lambda texts: [_stub_prob(t) for t in texts]
        elif head:
            self._prob = _deferred_backend(head)  # type: ignore[assignment]
        else:
            raise ValueError("OrmScorer needs head=<checkpoint> (real) or stub=true (plumbing smoke)")

    def score_sync(self, sample: Sample) -> ScoreResult:
        p = float(self._prob([_solution_view(sample)])[0])
        return ScoreResult(score=p, info={"orm_p_correct": p, "stub": self.stub})


class PrmScorer(Scorer):
    """Process reward: product of per-step ``P(step ok)``, read at each step boundary."""

    name = "prm"

    def __init__(
        self,
        head: str = "",
        stub: bool = False,
        agg: str = "product",
        device: str = "cuda",
        max_length: int = 2048,
    ):
        self.head_path = head
        self.stub = bool(stub)
        self.agg = agg
        self.device, self.max_length = device, int(max_length)
        if agg not in ("product", "min", "mean"):
            raise ValueError(f"prm agg must be product|min|mean, got {agg!r}")
        if self.stub:
            #: step-text -> P(step ok). The real backend reads the step-boundary
            #: token of each step from one forward pass over the whole solution;
            #: the stub scores each step's text independently, which is enough to
            #: exercise the aggregation but is NOT how the trained PRM works.
            self._step_probs: Callable[[str], list[float]] = lambda completion: [
                _stub_prob(s) for s in _steps(completion)
            ]
        elif head:
            self._step_probs = _deferred_backend(head)  # type: ignore[assignment]
        else:
            raise ValueError("PrmScorer needs head=<checkpoint> (real) or stub=true (plumbing smoke)")

    def score_sync(self, sample: Sample) -> ScoreResult:
        probs = [float(p) for p in self._step_probs(sample.completion)]
        if not probs:
            # A completion with no discernible step gets the lowest score rather
            # than a neutral default: under group normalisation a middling default
            # is a systematic bias, and an unparseable solution should not be safe.
            return ScoreResult(score=0.0, info={"prm_n_steps": 0, "stub": self.stub})
        product = math.exp(sum(math.log(max(p, 1e-6)) for p in probs))  # log-space for stability
        chosen = {"product": product, "min": min(probs), "mean": sum(probs) / len(probs)}[self.agg]
        return ScoreResult(
            score=float(chosen),
            info={
                "prm_n_steps": len(probs),
                "prm_product": product,
                "prm_min": min(probs),
                "prm_mean": sum(probs) / len(probs),
                "prm_agg": self.agg,
                "stub": self.stub,
            },
        )


register("orm", OrmScorer)
register("prm", PrmScorer)
