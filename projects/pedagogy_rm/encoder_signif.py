"""Is the gap between two encoders larger than the noise in measuring it?

    python projects/pedagogy_rm/encoder_signif.py --a olmo7b --b qwen7b

The sweep reports one number per encoder per dimension and they land within 0.01-0.03 of each other.
That is small enough that the ranking could be an artefact of which questions fell in which fold, so
this bootstraps the difference instead of trusting the point estimates.

PAIRED, AND RESAMPLED BY QUESTION. Both encoders score the same 600 turns, so the comparison is
paired and the bootstrap has to preserve that: each replicate draws questions with replacement and
takes both encoders' predictions for the turns in those questions. Resampling turns independently
would break the pairing and also ignore that turns from one question are correlated, which would
make every interval too narrow.

The interval is on r(b) - r(a). If it straddles zero, the encoders are not distinguishable on this
data and the choice should be made on cost instead.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="olmo7b", help="baseline tag under data/encoders")
    parser.add_argument("--b", default="qwen7b", help="challenger tag")
    parser.add_argument("--dimensions", default="leak,targeted,actionable,elicits")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", default="data/labels/*.json")
    parser.add_argument(
        "--cells",
        default="olmo7b:leak=last:16,targeted=mean:16,actionable=last:20,elicits=last:16;"
        "qwen7b:leak=eot:21,targeted=eot:18,actionable=last:18,elicits=eot:14",
        help="tag:dim=pooling:layer,...;tag:... . Read off the sweep, so the search is not redone: "
        "re-finding the best cell costs 3 poolings x ~30 layers x 5 folds of RidgeCV on 4096 "
        "dimensions per encoder per dimension, which is minutes of work to recover a number that "
        "is already printed in the sweep log.",
    )
    parser.add_argument("--slices", default="data/label_slices/slice_*.json")
    args = parser.parse_args()

    import numpy as np  # noqa: PLC0415
    from sklearn.linear_model import RidgeCV  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    from projects.pedagogy_rm.agreement import load as load_labels  # noqa: PLC0415
    from projects.pedagogy_rm.agreement import pearson  # noqa: PLC0415
    from projects.pedagogy_rm.probe import consensus, folds  # noqa: PLC0415

    by_unit = load_labels(sorted(set(itertools.chain.from_iterable(
        glob.glob(p) or [p] for p in [args.labels]))))
    question = {}
    for path in sorted(glob.glob(args.slices)):
        with open(path) as handle:
            for u in json.load(handle).get("units", []):
                question[u["id"]] = u.get("question", u["id"])

    blobs = {tag: np.load(f"data/encoders/{tag}.npz", allow_pickle=False) for tag in (args.a, args.b)}
    ids = {tag: [str(x) for x in blobs[tag]["ids"]] for tag in blobs}
    layers = {tag: list(blobs[tag]["layers"]) for tag in blobs}

    cellmap: dict[str, dict[str, tuple[str, int]]] = {}
    for block in args.cells.split(";"):
        tag, _, rest = block.partition(":")
        cellmap[tag.strip()] = {}
        for entry in rest.split(","):
            key, _, spec = entry.partition("=")
            pooling, _, layer = spec.partition(":")
            cellmap[tag.strip()][key.strip()] = (pooling.strip(), int(layer))

    def oof(tag, dim, usable):
        """Out-of-fold predictions at the cell the sweep already chose for this encoder."""
        index = {u: i for i, u in enumerate(ids[tag])}
        rows = np.array([index[u] for u in usable])
        y = np.array([consensus(by_unit, dim)[u] for u in usable], dtype=np.float32)
        groups = [question.get(u, u) for u in usable]
        pooling, layer = cellmap[tag][dim]
        li = layers[tag].index(layer)
        X = blobs[tag][pooling].astype(np.float32)[rows, li, :]
        pred = np.zeros(len(y))
        for test in folds(groups, args.folds, args.seed):
            tr = [i for i in range(len(y)) if i not in set(test)]
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(X[tr]), y[tr])
            pred[test] = m.predict(sc.transform(X[test]))
        return pred, y, groups, pooling, layer, pearson(list(map(float, pred)), list(map(float, y)))

    rng = np.random.default_rng(args.seed)
    print(f"paired bootstrap over questions, {args.boot} replicates\n")
    print(f"{'dimension':<12}{args.a:>9}{args.b:>9}{'diff':>8}{'95% interval':>20}{'':>4}")
    for dim in args.dimensions.split(","):
        scores = consensus(by_unit, dim)
        usable = [u for u in scores if u in {i for t in ids for i in ids[t]} and all(u in ids[t] for t in ids)]
        usable = [u for u in usable if u in question]
        if len(usable) < 40:
            print(f"{dim:<12} too few units ({len(usable)})")
            continue
        pa, y, groups, poola, la, ra = oof(args.a, dim, usable)
        pb, _, _, poolb, lb, rb = oof(args.b, dim, usable)

        qs = np.array(groups)
        uniq = np.unique(qs)
        by_q = {q: np.where(qs == q)[0] for q in uniq}
        diffs = []
        for _ in range(args.boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([by_q[q] for q in pick])
            diffs.append(pearson(list(map(float, pb[idx])), list(map(float, y[idx])))
                         - pearson(list(map(float, pa[idx])), list(map(float, y[idx]))))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        verdict = "distinguishable" if lo > 0 or hi < 0 else "not distinguishable"
        print(f"{dim:<12}{ra:>9.3f}{rb:>9.3f}{rb - ra:>+8.3f}   [{lo:>+6.3f}, {hi:>+6.3f}]  {verdict}")
        print(f"{'':<12}  cells: {args.a} {poola}/{la}, {args.b} {poolb}/{lb}")


if __name__ == "__main__":
    main()
