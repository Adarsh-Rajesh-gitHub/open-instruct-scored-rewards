"""Turning four CLI scalars into vLLM's ``speculative_config`` dict.

Scalars rather than one JSON blob, for two reasons. The platform submits a run as a single
command string inside a JSON payload inside a ``bash -lc '...'``, and a nested JSON dict does
not survive that without quoting no one can review. And the four knobs below are exactly the
ones arXiv:2604.26779 varies -- method, draft checkpoint, draft length k, drafter TP -- so
naming them individually keeps a run's command readable as the experiment it is.

Imports no vLLM, so it can be unit-tested on a machine that has none.
"""

from __future__ import annotations

from typing import Any

#: Draft length. The paper's Table 4 is unambiguous that this is not a "higher is better" knob:
#: on RL-Zero acceptance rises from 3.32 to 5.06 between k=3 and k=7 while realised speedup
#: *falls* from 1.77x to 1.21x, and on RL-Think k>=5 is slower than plain autoregressive decode.
#: k=3 is their best end-to-end setting in every configuration they report.
DEFAULT_NUM_SPECULATIVE_TOKENS = 3

SUPPORTED_METHODS = ("eagle3",)


def build_speculative_config(
    method: str | None,
    model: str | None,
    num_speculative_tokens: int = DEFAULT_NUM_SPECULATIVE_TOKENS,
    draft_tensor_parallel_size: int | None = None,
) -> dict[str, Any] | None:
    """Build the dict vLLM's ``AsyncEngineArgs`` takes as ``speculative_config``.

    Returns ``None`` when ``method`` is unset, which is what keeps an unflagged run
    byte-identical to upstream: ``speculative_config=None`` is ``AsyncEngineArgs``' own default.

    :raises ValueError: if the flags are set to a combination that cannot work. Raised here,
        from the driver, rather than surfacing later as an engine that came up without a
        drafter -- speculative decoding failing open is invisible in the metrics that matter,
        because a rollout with no drafting is still a *correct* rollout, only a slower one.
    """
    if method is None:
        if model is not None:
            raise ValueError(
                "--vllm_speculative_model was given without --vllm_speculative_method. "
                "The draft would be ignored and the rollout would run autoregressively."
            )
        return None

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported --vllm_speculative_method {method!r}; expected one of {SUPPORTED_METHODS}")

    # n-gram is deliberately absent from SUPPORTED_METHODS rather than merely untested. The
    # paper measures it at 0.7x on RL-Zero and 0.5x on RL-Think -- slower than no speculation
    # at all, despite acceptance lengths of 2.47 and 2.05 -- because verification overhead
    # erases the benefit. It is a trap worth naming: positive acceptance is not enough.
    if not model:
        raise ValueError(
            f"--vllm_speculative_method {method!r} needs --vllm_speculative_model pointing at a "
            "trained draft checkpoint. There is no pretrained EAGLE-3 draft for OLMoE to fall "
            "back on."
        )

    if num_speculative_tokens < 1:
        raise ValueError(f"--vllm_num_speculative_tokens must be >= 1, got {num_speculative_tokens}")

    config: dict[str, Any] = {"method": method, "model": model, "num_speculative_tokens": num_speculative_tokens}
    if draft_tensor_parallel_size is not None:
        config["draft_tensor_parallel_size"] = draft_tensor_parallel_size
    return config
