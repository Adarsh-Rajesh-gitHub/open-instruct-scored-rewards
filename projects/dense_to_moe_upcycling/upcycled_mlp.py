"""Function-preserving sparse expert wrappers for Qwen MLP blocks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class UpcycledQwenMLP(nn.Module):
    """Clone a dense Qwen MLP into top-k routed experts.

    At initialization every expert is identical and selected routing weights
    sum to one, so this module computes the same function as the dense MLP.
    """

    def __init__(
        self,
        dense_mlp: nn.Module,
        num_experts: int = 4,
        top_k: int = 2,
        router_std: float = 0.02,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least two")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if not hasattr(dense_mlp, "gate_proj"):
            raise TypeError("dense_mlp must expose a Qwen-style gate_proj")

        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [copy.deepcopy(dense_mlp) for _ in range(num_experts)]
        )
        reference = dense_mlp.gate_proj.weight
        hidden_size = dense_mlp.gate_proj.in_features
        self.router = nn.Linear(
            hidden_size,
            num_experts,
            bias=True,
            device=reference.device,
            dtype=reference.dtype,
        )
        nn.init.normal_(self.router.weight, mean=0.0, std=router_std)
        nn.init.zeros_(self.router.bias)
        self.register_buffer(
            "cumulative_counts",
            torch.zeros(
                num_experts,
                dtype=torch.long,
                device=reference.device,
            ),
            persistent=False,
        )
        self._last_router_logits: torch.Tensor | None = None
        self._last_top_indices: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        router_logits = self.router(flat).float()
        top_logits, top_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        weights = torch.softmax(top_logits, dim=-1)
        if self.top_k == 1:
            weights = torch.ones_like(weights)
        else:
            weights = torch.cat(
                (
                    weights[:, :-1],
                    1.0 - weights[:, :-1].sum(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
        output = torch.zeros_like(flat, dtype=torch.float32)

        for expert_index, expert in enumerate(self.experts):
            token_indices, slots = torch.where(
                top_indices == expert_index
            )
            if token_indices.numel() == 0:
                continue
            expert_output = expert(flat[token_indices])
            weighted = expert_output.float() * weights[
                token_indices, slots, None
            ]
            output = output.index_add(0, token_indices, weighted)

        with torch.no_grad():
            counts = torch.bincount(
                top_indices.reshape(-1), minlength=self.num_experts
            )
            self.cumulative_counts.add_(counts)
        self._last_router_logits = router_logits
        self._last_top_indices = top_indices
        return output.to(flat.dtype).reshape(shape)

    def router_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_router_logits is None or self._last_top_indices is None:
            raise RuntimeError("router losses require a preceding forward pass")
        probabilities = torch.softmax(self._last_router_logits, dim=-1)
        top_one = torch.nn.functional.one_hot(
            self._last_top_indices[:, 0],
            num_classes=self.num_experts,
        ).to(probabilities.dtype)
        load_balance = self.num_experts * torch.sum(
            probabilities.mean(dim=0) * top_one.mean(dim=0)
        )
        z_loss = torch.mean(
            torch.logsumexp(self._last_router_logits, dim=-1).square()
        )
        return load_balance, z_loss

    def reset_routing_counts(self) -> None:
        self.cumulative_counts.zero_()

    def routing_fractions(self) -> list[float]:
        total = int(self.cumulative_counts.sum().item())
        if total == 0:
            return [0.0] * self.num_experts
        return [
            float(value / total)
            for value in self.cumulative_counts.tolist()
        ]


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    if spec == "last2":
        return list(range(max(0, num_layers - 2), num_layers))
    if spec == "last4":
        return list(range(max(0, num_layers - 4), num_layers))
    indices = sorted({int(value.strip()) for value in spec.split(",")})
    if not indices or indices[0] < 0 or indices[-1] >= num_layers:
        raise ValueError(
            f"layer indices must be in [0, {num_layers - 1}]"
        )
    return indices


def upcycle_qwen_layers(
    model: nn.Module,
    layer_indices: Iterable[int],
    num_experts: int = 4,
    top_k: int = 2,
    router_std: float = 0.02,
) -> dict:
    layers = model.model.layers
    indices = sorted(set(layer_indices))
    for index in indices:
        if isinstance(layers[index].mlp, UpcycledQwenMLP):
            raise ValueError(f"layer {index} is already upcycled")
        layers[index].mlp = UpcycledQwenMLP(
            layers[index].mlp,
            num_experts=num_experts,
            top_k=top_k,
            router_std=router_std,
        )
    return {
        "layer_indices": indices,
        "num_experts": num_experts,
        "top_k": top_k,
        "router_std": router_std,
    }


def iter_upcycled_layers(model: nn.Module):
    for index, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, UpcycledQwenMLP):
            yield index, layer.mlp


def combined_router_loss(
    model: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = [module.router_losses() for _, module in iter_upcycled_layers(model)]
    if not losses:
        raise ValueError("model has no upcycled layers")
    load = torch.stack([pair[0] for pair in losses]).mean()
    z_loss = torch.stack([pair[1] for pair in losses]).mean()
    return load, z_loss


def routing_report(model: nn.Module) -> dict[str, list[float]]:
    return {
        str(index): module.routing_fractions()
        for index, module in iter_upcycled_layers(model)
    }


def set_only_moe_trainable(
    model: nn.Module,
    train_experts: bool = True,
    train_router: bool = True,
) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, module in iter_upcycled_layers(model):
        for parameter in module.experts.parameters():
            parameter.requires_grad_(train_experts)
        for parameter in module.router.parameters():
            parameter.requires_grad_(train_router)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def save_moe_checkpoint(
    model: nn.Module,
    path: Path,
    base_model: str,
    manifest: dict,
    extra: dict | None = None,
) -> None:
    modules = {
        str(index): module.state_dict()
        for index, module in iter_upcycled_layers(model)
    }
    payload = {
        "base_model": base_model,
        "manifest": manifest,
        "moe_modules": modules,
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "base_model": base_model,
                "manifest": manifest,
                "extra": extra or {},
            },
            indent=2,
        )
        + "\n"
    )


def restore_moe_checkpoint(model: nn.Module, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = payload["manifest"]
    upcycle_qwen_layers(
        model,
        manifest["layer_indices"],
        num_experts=manifest["num_experts"],
        top_k=manifest["top_k"],
        router_std=manifest.get("router_std", 0.02),
    )
    modules = dict(iter_upcycled_layers(model))
    for index, state in payload["moe_modules"].items():
        modules[int(index)].load_state_dict(state)
    return payload
