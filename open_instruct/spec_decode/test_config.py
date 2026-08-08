"""Tests for the speculative-decoding config builder.

Imports no vLLM, so this runs on a laptop. The registration and model-class tests need vLLM
and live in ``test_olmoe_eagle3.py``, which ``conftest.py`` skips when it is absent.
"""

from __future__ import annotations

import pytest

from open_instruct.spec_decode.config import DEFAULT_NUM_SPECULATIVE_TOKENS, build_speculative_config

DRAFT = "/checkpoints/olmoe-eagle3-draft"


class TestUnflaggedIsUnchanged:
    def test_no_method_returns_none(self):
        # The claim the whole patch rests on: an unflagged run passes speculative_config=None,
        # which is AsyncEngineArgs' own default, so the engine is built exactly as before.
        assert build_speculative_config(method=None, model=None) is None

    def test_default_k_is_three(self):
        # Paper Table 4: k=3 is the best end-to-end setting in every configuration reported,
        # even though acceptance keeps rising to k=7.
        assert DEFAULT_NUM_SPECULATIVE_TOKENS == 3
        built = build_speculative_config(method="eagle3", model=DRAFT)
        assert built["num_speculative_tokens"] == 3


class TestBuild:
    def test_minimal_config(self):
        assert build_speculative_config(method="eagle3", model=DRAFT) == {
            "method": "eagle3",
            "model": DRAFT,
            "num_speculative_tokens": 3,
        }

    def test_draft_tp_is_omitted_unless_set(self):
        # Absent rather than None, so vLLM applies its own default instead of being handed one.
        assert "draft_tensor_parallel_size" not in build_speculative_config(method="eagle3", model=DRAFT)
        built = build_speculative_config(method="eagle3", model=DRAFT, draft_tensor_parallel_size=1)
        assert built["draft_tensor_parallel_size"] == 1

    @pytest.mark.parametrize("k", [1, 3, 5, 7])
    def test_sweep_values_of_k_are_accepted(self, k):
        assert (
            build_speculative_config(method="eagle3", model=DRAFT, num_speculative_tokens=k)["num_speculative_tokens"]
            == k
        )


class TestFailsClosed:
    """Every case here would otherwise produce a rollout that is correct but not accelerated.

    That is the failure mode worth being loud about: speculative decoding cannot change what
    is sampled, so a misconfiguration does not corrupt a run -- it silently produces the
    baseline while the logs say speculative decoding was requested. Which is exactly the
    result the experiment is trying to measure.
    """

    def test_model_without_method_is_rejected(self):
        with pytest.raises(ValueError, match="without --vllm_speculative_method"):
            build_speculative_config(method=None, model=DRAFT)

    def test_method_without_model_is_rejected(self):
        with pytest.raises(ValueError, match="needs --vllm_speculative_model"):
            build_speculative_config(method="eagle3", model=None)

    def test_empty_model_is_rejected(self):
        with pytest.raises(ValueError, match="needs --vllm_speculative_model"):
            build_speculative_config(method="eagle3", model="")

    def test_ngram_is_rejected(self):
        # Not merely unimplemented. Paper Table 2 measures n-gram drafting at 0.7x on RL-Zero
        # and 0.5x on RL-Think -- slower than no speculation -- despite acceptance lengths of
        # 2.47 and 2.05. Accepting it here would invite a run that is worse than the baseline.
        with pytest.raises(ValueError, match="unsupported --vllm_speculative_method"):
            build_speculative_config(method="ngram", model=DRAFT)

    @pytest.mark.parametrize("k", [0, -1])
    def test_nonpositive_k_is_rejected(self, k):
        with pytest.raises(ValueError, match="must be >= 1"):
            build_speculative_config(method="eagle3", model=DRAFT, num_speculative_tokens=k)
