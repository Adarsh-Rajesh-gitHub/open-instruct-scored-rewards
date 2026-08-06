"""Pull the full W&B history for every arm into one file.

    # on a machine that can reach wandb (the cluster can):
    python projects/pedagogy_rm/fetch_history.py --out data/allhist.json

ONE FILE FOR EVERY ARM, because the arms live in two entities. A and B were logged before
WANDB_ENTITY was set and sit under the personal account; everything from C onwards is under
eduLLM. Anything drawing a figure across all seven would otherwise need credentials for both and
would break differently depending on which one it could reach.

THREE CADENCES FETCHED SEPARATELY, WHICH IS FORCED RATHER THAN CHOSEN. scan_history returns only
those rows that carry every key requested, so mixing per-step keys with eval keys returns their
intersection - a handful of rows out of two hundred - and a loss curve drawn from that looks like a
run that never trained. Optimiser keys are split off for the same reason: they are absent from the
first row of some runs, which would truncate the reward series if fetched alongside it.
"""

from __future__ import annotations

import argparse
import json

MIT = "zsophia-massachusetts-institute-of-technology/pedagogy-rm"
EDU = "eduLLM/pedagogy-rm"

# label -> (path, prefix for the scorer's own metrics). The prefix differs because arm B ran the
# z-scored scorer, which registers under its own name.
RUNS = {
    "A": (f"{MIT}/pm48fp8k", "pedagogy"),
    "B": (f"{MIT}/k77lk6tm", "pedagogy_z"),
    "C": (f"{EDU}/m5n6r18u", "pedagogy"),
    "D": (f"{EDU}/ociljku8", "pedagogy"),
    "E": (f"{EDU}/118zudzw", "pedagogy"),
    # Arm F is three W&B runs because the partition preempted it twice and Slurm requeued it; each
    # requeue resumes from the last checkpoint but starts a new run. Its curve has to be stitched
    # from all three or it appears to begin at step 71, which reads as a shorter experiment rather
    # than as an interrupted one. The two earlier ones are fetched under their own labels and
    # joined when plotting, so the join is visible in one place rather than hidden in the dump.
    "F": (f"{EDU}/crz0ywl5", "pedagogy"),
    "F_try1": (f"{EDU}/qnlwwreo", "pedagogy"),
    "F_try2": (f"{EDU}/uyi7t6nn", "pedagogy"),
    "G": (f"{EDU}/0gox5m1k", "pedagogy"),
    "lr2": (f"{EDU}/n673jwne", "pedagogy"),
    "lr4": (f"{EDU}/ovgw2n3a", "pedagogy"),
    "lr8": (f"{EDU}/lrzqocry", "pedagogy"),
}

DIMS = ("targeted", "actionable", "elicits", "leak", "correct", "length")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/allhist.json")
    args = parser.parse_args()

    import wandb  # noqa: PLC0415

    api = wandb.Api()
    blob = {}
    for label, (path, prefix) in RUNS.items():
        try:
            run = api.run(path)
        except Exception as exc:  # noqa: BLE001 - a missing run should not lose the other nine
            print(f"{label}: cannot read {path} ({str(exc)[:60]})")
            continue
        have = set(run.summary.keys())

        def keep(keys: list[str], have: set = have) -> list[str]:
            return ["_step"] + [k for k in keys if k in have]

        step = keep(["scores", "objective/kl1_avg", "loss/policy_avg", "loss/kl_avg",
                     "loss/total_avg", "lr", f"scored/{prefix}/reward",
                     f"scored/{prefix}/words", f"scored/{prefix}/zero_advantage"]
                    + [f"scored/{prefix}/dim_{d}" for d in DIMS])
        optim = keep(["optim/grad_norm", "policy/clipfrac_avg", "val/ratio_var",
                      "val/advantages_mean", "val/stop_rate", "val/sequence_lengths"])
        ev = keep(["eval/scores", "eval/sequence_lengths", f"eval/scored/{prefix}/reward",
                   f"eval/scored/{prefix}/words"]
                  + [f"eval/scored/{prefix}/dim_{d}" for d in DIMS])

        blob[label] = {
            "prefix": prefix,
            "config": {k: v for k, v in run.config.items()
                       if k in ("learning_rate", "num_samples_per_prompt_rollout", "beta",
                                "num_unique_prompts_rollout", "total_episodes", "exp_name")},
            "state": run.state,
            "step": list(run.scan_history(keys=step, page_size=2000)),
            "optim": list(run.scan_history(keys=optim, page_size=2000)),
            "eval": list(run.scan_history(keys=ev, page_size=2000)),
        }
        print(f"{label}: {len(blob[label]['step'])} step rows, "
              f"{len(blob[label]['optim'])} optim, {len(blob[label]['eval'])} eval")

    with open(args.out, "w") as handle:
        json.dump(blob, handle)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
