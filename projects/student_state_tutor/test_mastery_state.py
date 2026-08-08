import pytest

from projects.student_state_tutor.mastery_state import (
    ConceptBelief,
    EvidenceEvent,
    MasteryGraphState,
)


def event(concept_id, correct, assistance="unaided"):
    return EvidenceEvent(
        learner_id="student",
        concept_id=concept_id,
        item_id=f"{concept_id}-item",
        correct=correct,
        assistance=assistance,
        timestamp="2026-08-06T00:00:00+00:00",
    )


def test_unaided_evidence_updates_beta_posterior():
    graph = MasteryGraphState("student", ["forces"])
    graph.observe(event("forces", True))
    graph.observe(event("forces", False))
    belief = graph.belief("forces")
    assert belief.alpha == 2.0
    assert belief.beta == 2.0
    assert belief.mean == 0.5
    assert belief.unaided_n == 2


def test_revealed_answer_does_not_change_mastery():
    graph = MasteryGraphState("student", ["forces"])
    graph.observe(event("forces", True, assistance="revealed"))
    belief = graph.belief("forces")
    assert belief.alpha == 1.0
    assert belief.beta == 1.0
    assert belief.assisted_n == 1
    assert belief.effective_evidence == 0.0


def test_graph_selects_weakest_and_uncertain_nodes():
    graph = MasteryGraphState("student", ["weak", "strong", "unknown"])
    graph.observe(event("weak", False))
    graph.observe(event("weak", False))
    graph.observe(event("strong", True))
    graph.observe(event("strong", True))
    assert graph.weakest() == "weak"
    assert graph.most_informative() == "unknown"


def test_snapshot_round_trip_is_deterministic():
    graph = MasteryGraphState("student", ["forces"])
    graph.observe(event("forces", False))
    restored = MasteryGraphState.restore(graph.snapshot())
    assert restored.snapshot() == graph.snapshot()
    assert restored.to_json() == graph.to_json()


def test_event_rejects_unknown_assistance():
    belief = ConceptBelief("forces")
    with pytest.raises(ValueError, match="unknown assistance"):
        belief.update(event("forces", True, assistance="magic"))
