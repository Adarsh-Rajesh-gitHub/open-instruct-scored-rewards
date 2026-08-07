"""Plumbing smoke for the PRM-vs-ORM reward plugin.

    python projects/prm_vs_orm/selftest.py

This is the payload of the first platform submission (the ``-check`` workload,
``--dataset none``, no GPU model, no checkpoint). It proves the things that can
only be proven on the platform image itself:

  * our committed tree is in the image and imports under its Python;
  * ``registry.load_plugins`` finds ``projects/prm_vs_orm/plugin.py`` and importing
    it registers ``orm`` and ``prm`` by side effect;
  * ``registry.build`` resolves both names -- the exact call ``grpo_fast`` makes
    for ``--group_scorer`` -- and the returned scorer honours the ``Scorer`` /
    ``GroupScorer`` contract on a synthetic group;
  * the PRM's product aggregation is arithmetically what it claims to be.

It deliberately uses the ``stub=true`` backend, so it needs neither a reward-model
checkpoint nor a GPU. It asserts, prints ``PLUMBING OK``, and exits 0; any failure
raises and exits non-zero, which is what the platform reads as a failed check.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys

# Run as a plain script (``python projects/prm_vs_orm/selftest.py``), sys.path[0] is
# this file's directory, not the repo root, so ``open_instruct`` would not import. The
# image does not pip-install the package; it relies on the repo root being the cwd. Put
# the repo root (two levels up from projects/prm_vs_orm/) on the path so the script runs
# the same way whether launched by path, by ``-m``, or from another directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from open_instruct.scored_rewards import Sample, registry  # noqa: E402

#: Anchored to the repo root, not cwd, so the smoke passes however it is launched.
#: grpo_fast passes this same repo-relative path from the repo root, where it resolves
#: identically.
PLUGIN = os.path.join(_REPO_ROOT, "projects", "prm_vs_orm", "plugin.py")


def _group() -> list[Sample]:
    """A handful of synthetic solutions, one per prompt does not matter here."""
    return [
        Sample(
            prompt="What is 2 + 3?\n\n",
            completion="Add the two numbers.\n\nTwo plus three is five.\n\n# Answer\n\n5",
        ),
        Sample(
            prompt="What is 10 - 4?\n\n",
            completion="Subtract four from ten.\n\n# Answer\n\n6",
        ),
        Sample(prompt="What is 7 * 6?\n\n", completion="# Answer\n\n42"),
    ]


def main() -> None:
    # 1. The import path grpo_fast uses. Importing the plugin registers the scorers.
    loaded = registry.load_plugins(PLUGIN)
    assert loaded == [PLUGIN], f"expected to load {PLUGIN!r}, loaded {loaded!r}"
    available = registry.available()
    for name in ("orm", "prm"):
        assert name in available, f"{name!r} did not register; registry has {available!r}"

    group = _group()

    # 2. ORM: build via the real spec path, score the group, check the contract.
    orm = registry.build("orm:stub=true")
    orm_results = asyncio.run(orm.score_group(group))
    assert len(orm_results) == len(group)
    for r in orm_results:
        assert 0.0 <= r.score <= 1.0, f"orm score out of range: {r.score}"
        assert r.info.get("stub") is True

    # 3. PRM: same, and verify the product aggregation is exactly the product of
    #    the per-step probabilities reported in info.
    prm = registry.build("prm:stub=true")
    prm_results = asyncio.run(prm.score_group(group))
    assert len(prm_results) == len(group)
    for r in prm_results:
        assert 0.0 <= r.score <= 1.0, f"prm score out of range: {r.score}"
        n = r.info["prm_n_steps"]
        assert n >= 1, f"expected at least one step, got {n}"
        # product must equal min <= mean bounds and match a fresh recomputation
        assert r.info["prm_min"] <= r.info["prm_mean"] + 1e-9
        assert r.info["prm_product"] <= r.info["prm_min"] + 1e-9  # product of <=1 terms <= any single term
        assert math.isfinite(r.info["prm_product"])
        assert abs(r.score - r.info["prm_product"]) < 1e-9  # default agg is product

    # 4. min/mean aggregations are selectable and differ from product on >1 step.
    prm_min = registry.build("prm:stub=true,agg=min")
    min_results = asyncio.run(prm_min.score_group(group))
    multi = [i for i, r in enumerate(prm_results) if r.info["prm_n_steps"] > 1]
    assert multi, "no multi-step sample to distinguish product from min"
    i = multi[0]
    assert abs(min_results[i].score - prm_results[i].info["prm_min"]) < 1e-9

    print(
        "PLUMBING OK: "
        f"orm+prm registered, built and scored {len(group)} samples; "
        f"prm steps={[r.info['prm_n_steps'] for r in prm_results]}"
    )


if __name__ == "__main__":
    main()
