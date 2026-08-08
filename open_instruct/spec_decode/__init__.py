"""Speculative decoding for RL rollouts (fork addition).

Nothing here is imported by upstream open-instruct. The entry point is
``register_olmoe_eagle3`` in :mod:`open_instruct.spec_decode.olmoe_eagle3`, called from
``vllm_utils.LLMRayActor`` before the engine is built when ``--speculative_config`` names
a draft. With that flag unset nothing in this package is imported at all.
"""
