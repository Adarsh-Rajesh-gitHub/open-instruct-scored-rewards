"""Sparse Bayesian mastery state owned by the tutoring environment."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

ASSISTANCE_RELIABILITY = {"unaided": 1.0, "hinted": 0.35, "worked": 0.1, "revealed": 0.0}


@dataclass(frozen=True)
class EvidenceEvent:
    learner_id: str
    concept_id: str
    item_id: str
    correct: bool
    assistance: str = "unaided"
    reliability: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "external_grader"

    @property
    def weight(self) -> float:
        if self.assistance not in ASSISTANCE_RELIABILITY:
            raise ValueError(f"unknown assistance level: {self.assistance}")
        return max(0.0, self.reliability) * ASSISTANCE_RELIABILITY[self.assistance]


@dataclass
class ConceptBelief:
    concept_id: str
    alpha: float = 1.0
    beta: float = 1.0
    unaided_n: int = 0
    assisted_n: int = 0
    effective_evidence: float = 0.0
    last_observed_at: str | None = None

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        total = self.alpha + self.beta
        return self.alpha * self.beta / (total * total * (total + 1.0))

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(self.variance)

    @property
    def interval_approx(self) -> tuple[float, float]:
        radius = 1.96 * self.standard_deviation
        return max(0.0, self.mean - radius), min(1.0, self.mean + radius)

    def update(self, event: EvidenceEvent) -> None:
        if event.concept_id != self.concept_id:
            raise ValueError("event concept does not match belief")
        weight = event.weight
        if event.correct:
            self.alpha += weight
        else:
            self.beta += weight
        if event.assistance == "unaided":
            self.unaided_n += 1
        else:
            self.assisted_n += 1
        self.effective_evidence += weight
        self.last_observed_at = event.timestamp

    def expected_variance_after_one(self) -> float:
        probability_correct = self.mean
        correct = ConceptBelief(self.concept_id, alpha=self.alpha + 1.0, beta=self.beta)
        incorrect = ConceptBelief(self.concept_id, alpha=self.alpha, beta=self.beta + 1.0)
        return probability_correct * correct.variance + (1.0 - probability_correct) * incorrect.variance

    @property
    def information_gain(self) -> float:
        return self.variance - self.expected_variance_after_one()

    def to_view(self) -> dict:
        low, high = self.interval_approx
        return {
            "concept_id": self.concept_id,
            "mean": round(self.mean, 4),
            "interval": [round(low, 4), round(high, 4)],
            "unaided_n": self.unaided_n,
            "assisted_n": self.assisted_n,
            "effective_evidence": round(self.effective_evidence, 3),
            "last_observed_at": self.last_observed_at,
        }


class MasteryGraphState:
    """Sparse per-learner beliefs plus immutable evidence provenance."""

    def __init__(
        self, learner_id: str, concept_ids: Iterable[str] = (), prior_alpha: float = 1.0, prior_beta: float = 1.0
    ):
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta priors must be positive")
        self.learner_id = learner_id
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.beliefs = {
            concept_id: ConceptBelief(concept_id, alpha=prior_alpha, beta=prior_beta) for concept_id in concept_ids
        }
        self.events: list[EvidenceEvent] = []

    def belief(self, concept_id: str) -> ConceptBelief:
        if concept_id not in self.beliefs:
            self.beliefs[concept_id] = ConceptBelief(concept_id, alpha=self.prior_alpha, beta=self.prior_beta)
        return self.beliefs[concept_id]

    def observe(self, event: EvidenceEvent) -> None:
        if event.learner_id != self.learner_id:
            raise ValueError("event learner does not match graph state")
        self.belief(event.concept_id).update(event)
        self.events.append(event)

    def weakest(self, eligible: Iterable[str] | None = None) -> str:
        candidates = list(eligible) if eligible is not None else list(self.beliefs)
        if not candidates:
            raise ValueError("no eligible concepts")
        return min(
            candidates,
            key=lambda concept_id: (self.belief(concept_id).mean, -self.belief(concept_id).variance, concept_id),
        )

    def most_informative(self, eligible: Iterable[str] | None = None) -> str:
        candidates = list(eligible) if eligible is not None else list(self.beliefs)
        if not candidates:
            raise ValueError("no eligible concepts")
        return max(
            candidates,
            key=lambda concept_id: (
                self.belief(concept_id).information_gain,
                self.belief(concept_id).variance,
                concept_id,
            ),
        )

    def local_view(self, concept_ids: Iterable[str] | None = None, max_nodes: int | None = None) -> dict:
        selected = list(concept_ids) if concept_ids is not None else list(self.beliefs)
        selected.sort(
            key=lambda concept_id: (self.belief(concept_id).mean, -self.belief(concept_id).variance, concept_id)
        )
        if max_nodes is not None:
            selected = selected[:max_nodes]
        return {
            "learner_id": self.learner_id,
            "nodes": [self.belief(concept_id).to_view() for concept_id in selected],
            "event_count": len(self.events),
        }

    def snapshot(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "beliefs": {concept_id: asdict(belief) for concept_id, belief in sorted(self.beliefs.items())},
            "events": [asdict(event) for event in self.events],
        }

    @classmethod
    def restore(cls, snapshot: dict) -> MasteryGraphState:
        graph = cls(
            learner_id=snapshot["learner_id"],
            prior_alpha=float(snapshot["prior_alpha"]),
            prior_beta=float(snapshot["prior_beta"]),
        )
        graph.beliefs = {
            concept_id: ConceptBelief(**belief) for concept_id, belief in snapshot.get("beliefs", {}).items()
        }
        graph.events = [EvidenceEvent(**event) for event in snapshot.get("events", [])]
        return graph

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)
