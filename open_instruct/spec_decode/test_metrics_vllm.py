"""Tests for the vLLM stat logger that reads acceptance length out of the engine.

Needs vLLM, so ``conftest.py`` skips this file when it is absent. The pure-arithmetic half of
the telemetry is in ``test_reporting.py`` and runs anywhere.
"""

from __future__ import annotations

import pytest
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from open_instruct.spec_decode import metrics


class FakeSpeculativeConfig:
    def __init__(self, num_speculative_tokens: int):
        self.num_speculative_tokens = num_speculative_tokens


class FakeVllmConfig:
    def __init__(self, num_speculative_tokens: int | None = 3):
        self.speculative_config = (
            FakeSpeculativeConfig(num_speculative_tokens) if num_speculative_tokens is not None else None
        )


@pytest.fixture(autouse=True)
def clean_registry():
    metrics.reset_registry_for_testing()
    yield
    metrics.reset_registry_for_testing()


def scheduler_stats_with(num_spec_tokens: int, drafts: list[tuple[int, int]]) -> SchedulerStats:
    """A SchedulerStats carrying one iteration's worth of (draft_tokens, accepted) observations."""
    spec = SpecDecodingStats.new(num_spec_tokens)
    for num_draft_tokens, num_accepted in drafts:
        spec.observe_draft(num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted)
    return SchedulerStats(spec_decoding_stats=spec)


class TestContract:
    def test_satisfies_vllms_stat_logger_interface(self):
        # vLLM validates plugin loggers with issubclass; passing one directly to
        # from_engine_args skips that check, so assert it here instead.
        assert issubclass(metrics.SpecDecodeStatLogger, StatLoggerBase)

    def test_constructing_registers_the_instance(self):
        # The factory vLLM calls is the class itself, so a module-level registry is the only way
        # to reach the object it built.
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(), engine_index=0)
        assert metrics.drain_all() is not None
        assert logger.num_spec_tokens == 3

    def test_tolerates_a_config_with_no_speculative_section(self):
        # The baseline arm attaches this logger with no draft configured.
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(num_speculative_tokens=None))
        assert logger.num_spec_tokens == 0


class TestAccumulation:
    def test_records_and_drains(self):
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        # Two iterations, three drafts total: (3 proposed, 2 accepted) twice then (3, 2).
        logger.record(scheduler_stats_with(3, [(3, 2), (3, 2)]), None)
        logger.record(scheduler_stats_with(3, [(3, 2)]), None)
        drained = logger.drain()
        assert drained["num_drafts"] == 3
        assert drained["num_draft_tokens"] == 9
        assert drained["num_accepted_tokens"] == 6
        # vLLM's convention, bonus token included: 1 + 6/3.
        assert drained["acceptance_length"] == pytest.approx(3.0)
        assert drained["draft_acceptance_rate"] == pytest.approx(6 / 9)

    def test_drain_resets_so_the_series_is_per_step(self):
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        logger.record(scheduler_stats_with(3, [(3, 2)]), None)
        assert logger.drain()["num_drafts"] == 1
        second = logger.drain()
        assert second["num_drafts"] == 0
        assert second["acceptance_length"] is None

    def test_nothing_drafted_gives_none_not_one(self):
        # An acceptance length of 1.0 means "drafted, and every proposal was rejected" -- a real
        # and bad outcome. Not drafting at all is the baseline arm's normal state. Collapsing the
        # two would make a broken drafter indistinguishable from the baseline.
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        drained = logger.drain()
        assert drained["num_drafts"] == 0
        assert drained["acceptance_length"] is None

    def test_all_rejected_gives_exactly_one(self):
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        logger.record(scheduler_stats_with(3, [(3, 0), (3, 0)]), None)
        assert logger.drain()["acceptance_length"] == pytest.approx(1.0)

    def test_none_scheduler_stats_is_ignored(self):
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        logger.record(None, None)
        assert logger.drain()["num_drafts"] == 0

    def test_iteration_without_speculation_is_ignored(self):
        # Every iteration of the autoregressive arm takes this path.
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        logger.record(SchedulerStats(), None)
        assert logger.drain()["num_drafts"] == 0

    def test_per_position_acceptance_is_reported(self):
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        # Four drafts: two accept 3 tokens, one accepts 1, one accepts 0.
        logger.record(scheduler_stats_with(3, [(3, 3), (3, 3), (3, 1), (3, 0)]), None)
        per_pos = logger.drain()["per_position_acceptance"]
        # Position 0 survived 3 of 4 drafts; position 1 and 2 survived 2 of 4.
        assert per_pos == pytest.approx([0.75, 0.5, 0.5])


class TestDrainAll:
    def test_sums_counters_across_engines_before_computing_alpha(self):
        # A mean of per-engine alphas is not the alpha of the rollout unless both engines drafted
        # equally often. Engine 0: 1 draft, 0 accepted. Engine 1: 10 drafts, 30 accepted.
        # Summed: 1 + 30/11 = 3.727...  Averaged alphas would be (1.0 + 4.0)/2 = 2.5.
        first = metrics.SpecDecodeStatLogger(FakeVllmConfig(3), engine_index=0)
        second = metrics.SpecDecodeStatLogger(FakeVllmConfig(3), engine_index=1)
        first.record(scheduler_stats_with(3, [(3, 0)]), None)
        second.record(scheduler_stats_with(3, [(3, 3)] * 10), None)
        combined = metrics.drain_all()
        assert combined["num_drafts"] == 11
        assert combined["num_accepted_tokens"] == 30
        assert combined["acceptance_length"] == pytest.approx(1 + 30 / 11)

    def test_empty_registry_returns_empty(self):
        assert metrics.drain_all() == {}

    def test_drain_all_resets_every_engine(self):
        logger = metrics.SpecDecodeStatLogger(FakeVllmConfig(3))
        logger.record(scheduler_stats_with(3, [(3, 2)]), None)
        assert metrics.drain_all()["num_drafts"] == 1
        assert metrics.drain_all()["num_drafts"] == 0
