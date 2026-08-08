"""How much does per-expert LoRA cost, and which transformers representation is cheaper?

    python projects/pedagogy_rm/moe_bench.py --label venv_oi

Run once under each environment and compare. The two are not two ways of writing the same thing:

  transformers 4.x  experts are a ModuleList of 64 nn.Linear per layer, so PEFT builds
                    16 x 64 x 3 = 3072 separate LoRA modules and reaches them with target_modules.
  transformers 5.x  experts are two fused 3-D nn.Parameters per layer, so PEFT builds one stacked
                    adapter per parameter and reaches them with target_parameters. This is also the
                    layout vLLM's FusedMoEWithLoRA expects, so an adapter trained this way serves
                    without conversion.

PEFT's own documentation warns that expert LoRA "can add a substantial runtime overhead", because
the LoRA contribution is materialised for every expert even when only a few are routed to. One
published measurement put a vanilla implementation at >600% slower. That is a claim about a
different model and a different code path, so this measures ours rather than quoting theirs.

WHAT IS TIMED. A forward and a backward on one fixed batch, after warmup, with CUDA synchronised
either side - not wall time around a training step, which would fold in the data loader and the
optimiser. Peak allocated memory is read from torch, reset before the measured region.
"""

from __future__ import annotations

import argparse
import statistics
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    parser.add_argument("--label", required=True, help="what to call this environment in the output")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--expert-r", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    print(f"=== {args.label} | transformers {transformers.__version__} | torch {torch.__version__} ===")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load():
        # stubs read .to(str) as the module overload, hence the ignore
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)  # ty: ignore[invalid-argument-type]

    probe = load()
    mlp = probe.model.layers[0].mlp
    fused = not isinstance(getattr(mlp, "experts", None), torch.nn.ModuleList)
    print(f"expert layout: {'fused 3-D parameters' if fused else 'ModuleList of Linear'}")

    def wrap(model, moe: bool):
        if not moe:
            cfg = LoraConfig(
                r=args.r,
                lora_alpha=2 * args.r,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        elif fused:
            cfg = LoraConfig(
                r=args.r,
                lora_alpha=2 * args.r,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
                rank_pattern={"experts.gate_up_proj": args.expert_r, "experts.down_proj": args.expert_r},
            )
        else:
            # 4.x: the experts are real modules, so they are reachable by name. gate_proj/up_proj/
            # down_proj live inside every expert, so this creates one adapter per expert per layer.
            cfg = LoraConfig(
                r=args.expert_r,
                lora_alpha=2 * args.expert_r,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        return get_peft_model(model, cfg)

    ids = torch.randint(0, 1000, (args.batch, args.seq), device=dev)
    results = {}
    for tag, moe in (("attention only", False), ("attention + experts", True)):
        model = wrap(load(), moe)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_modules = sum(1 for n, _ in model.named_modules() if "lora_A" in n)
        model.train()
        times = []
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        for i in range(args.warmup + args.steps):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=ids, labels=ids).loss.backward()
            model.zero_grad(set_to_none=True)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        med = statistics.median(times)
        results[tag] = med
        print(
            f"  {tag:<22} {med * 1000:>8.1f} ms/step   {trainable / 1e6:>7.1f}M trainable   "
            f"{n_modules:>5} lora modules   peak {peak:>5.1f} GB"
        )
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    a, b = results["attention only"], results["attention + experts"]
    print(f"  -> per-expert LoRA costs {100 * (b - a) / a:+.0f}% per step in this environment")


if __name__ == "__main__":
    main()
