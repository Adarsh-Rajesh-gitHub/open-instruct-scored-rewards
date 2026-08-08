import copy

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from projects.dense_to_moe_upcycling.upcycled_mlp import (
    UpcycledQwenMLP,
    combined_router_loss,
    parse_layer_spec,
    restore_moe_checkpoint,
    save_moe_checkpoint,
    upcycle_qwen_layers,
)


def tiny_config():
    return Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


def test_cloned_experts_preserve_dense_mlp_function():
    torch.manual_seed(0)
    dense = Qwen2MLP(tiny_config()).eval()
    hidden = torch.randn(2, 7, 32)
    expected = dense(hidden)
    moe = UpcycledQwenMLP(dense, num_experts=4, top_k=2).eval()
    actual = moe(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert sum(moe.routing_fractions()) == 1.0


def test_upcycled_model_preserves_logits_and_router_has_gradients():
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(tiny_config())
    input_ids = torch.randint(0, 128, (2, 12))
    expected = model(input_ids).logits.detach()
    upcycle_qwen_layers(model, [1], num_experts=4, top_k=2)
    output = model(input_ids).logits
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    load_loss, z_loss = combined_router_loss(model)
    (output.square().mean() + 0.01 * load_loss + 0.001 * z_loss).backward()
    router = model.model.layers[1].mlp.router
    assert router.weight.grad is not None
    assert torch.isfinite(router.weight.grad).all()


def test_moe_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    original = Qwen2ForCausalLM(tiny_config())
    dense_state = copy.deepcopy(original.state_dict())
    manifest = upcycle_qwen_layers(
        original, [0, 1], num_experts=3, top_k=2
    )
    input_ids = torch.randint(0, 128, (1, 10))
    expected = original(input_ids).logits.detach()
    path = tmp_path / "moe.pt"
    save_moe_checkpoint(
        original,
        path,
        base_model="tiny",
        manifest=manifest,
        extra={"test": True},
    )

    restored = Qwen2ForCausalLM(tiny_config())
    restored.load_state_dict(dense_state)
    payload = restore_moe_checkpoint(restored, path)
    actual = restored(input_ids).logits.detach()
    torch.testing.assert_close(actual, expected)
    assert payload["extra"]["test"]


def test_parse_layer_spec():
    assert parse_layer_spec("last2", 6) == [4, 5]
    assert parse_layer_spec("last4", 6) == [2, 3, 4, 5]
    assert parse_layer_spec("1,3,3", 6) == [1, 3]
