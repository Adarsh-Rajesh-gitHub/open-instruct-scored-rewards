"""Registering OLMoE's EAGLE-3 target with vLLM, without importing it.

Deliberately separate from :mod:`open_instruct.spec_decode.olmoe_eagle3`, and the separation
is the entire point of the file. The registration is a *lazy* one -- a ``"<module>:<class>"``
string that vLLM imports in whichever process resolves the model -- so that torch's CUDA state
is not initialised in the parent, which is what would break the forked engine workers
open-instruct runs (``VLLM_ENABLE_V1_MULTIPROCESSING=0`` plus vLLM's default
``VLLM_WORKER_MULTIPROC_METHOD=fork``).

If the function that installs that string lived in the same module as the class, then importing
it in order to call it would import the class too, and the laziness would buy nothing. So this
module imports ``ModelRegistry`` and nothing else from vLLM.
"""

from __future__ import annotations

from vllm.model_executor.models.registry import ModelRegistry

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

#: The architecture string in ``allenai/OLMoE-1B-7B-0125-DPO``'s ``config.json``. Registering
#: under the same name is the point: vLLM resolves a model by the architecture its config
#: names, so overriding this key is what makes an unmodified checkpoint load our class.
OLMOE_ARCH = "OlmoeForCausalLM"

OLMOE_EAGLE3_MODULE = "open_instruct.spec_decode.olmoe_eagle3"
OLMOE_EAGLE3_CLASS = "OlmoeForCausalLMEagle3"
LAZY_TARGET = f"{OLMOE_EAGLE3_MODULE}:{OLMOE_EAGLE3_CLASS}"


def register_olmoe_eagle3() -> None:
    """Point ``OlmoeForCausalLM`` at the EAGLE-3-capable class. Idempotent.

    Call before the engine is built, in every process that will resolve the model.
    """
    if is_registered():
        return
    ModelRegistry.register_model(OLMOE_ARCH, LAZY_TARGET)
    logger.info("spec_decode: registered %s -> %s for EAGLE-3", OLMOE_ARCH, LAZY_TARGET)


def is_registered() -> bool:
    """True when the registry's entry for OLMoE is the lazy pointer this module installs."""
    entry = ModelRegistry.models.get(OLMOE_ARCH)
    return (
        getattr(entry, "module_name", None) == OLMOE_EAGLE3_MODULE
        and getattr(entry, "class_name", None) == OLMOE_EAGLE3_CLASS
    )


def assert_registered() -> None:
    """Raise unless ``OlmoeForCausalLM`` is registered to our class in *this* process.

    This is an early, specific check and not the only line of defence: vLLM does catch the
    case on its own, in ``gpu_model_runner``, with ``RuntimeError("Model does not support
    EAGLE3 interface but aux_hidden_state_outputs was requested")``. What that message cannot
    say is *why*, and it arrives from inside a worker after the engine has loaded a 7B
    checkpoint. Failing here instead costs a second and names the cause.

    The registry is read rather than resolved, because ``resolve_model_cls`` wants a
    ``ModelConfig`` this has no reason to hold, and resolving would import the class into a
    process that may not want CUDA touched yet.
    """
    if is_registered():
        return
    entry = ModelRegistry.models.get(OLMOE_ARCH)
    raise RuntimeError(
        f"{OLMOE_ARCH} is registered to "
        f"{getattr(entry, 'module_name', entry)}:{getattr(entry, 'class_name', '?')}, not "
        f"{LAZY_TARGET}. EAGLE-3 needs the auxiliary hidden states only the latter returns. "
        "Was register_olmoe_eagle3() called before the engine was built, in this process?"
    )
