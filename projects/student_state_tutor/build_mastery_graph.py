"""Build and audit a leak-resistant concept/prerequisite graph."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_annotations(path: Path) -> dict[str, dict]:
    return {
        normalize(row["key"]): row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def join_items(items_path: Path, annotations_path: Path, split: str) -> tuple[list[dict], int]:
    annotations = load_annotations(annotations_path)
    rows = []
    unmatched = 0
    for line in items_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("subject") == "math":
            continue
        key = normalize(item["question"])
        annotation = annotations.get(key)
        if annotation is None:
            unmatched += 1
            continue
        edges = [[normalize(source), normalize(target)] for source, target in annotation.get("prerequisites", [])]
        rows.append(
            {
                "key": key,
                "split": split,
                "subject": item.get("subject", "unknown"),
                "grade": item.get("grade"),
                "units": [normalize(unit) for unit in annotation.get("units", [])],
                "prerequisites": edges,
                "misconception": annotation.get("misconception", ""),
                "gold_text": item["choices"][item["gold_idx"]],
            }
        )
    return rows, unmatched


def connected_components(nodes: set[str], edges: Counter) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    components = []
    unseen = set(nodes)
    while unseen:
        root = unseen.pop()
        stack = [root]
        component = {root}
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unseen
            unseen -= neighbors
            component |= neighbors
            stack.extend(neighbors)
        components.append(sorted(component))
    return sorted(components, key=len, reverse=True)


def exact_transfer_pairs(rows: list[dict], left_split: str, right_split: str) -> list[dict]:
    by_unit: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for unit in row["units"]:
            by_unit[unit][row["split"]].append(row["key"])
    pairs = []
    for unit, split_rows in by_unit.items():
        if left_split == right_split:
            for left, right in combinations(sorted(set(split_rows[left_split])), 2):
                pairs.append({"unit": unit, "left": left, "right": right})
        else:
            for left in sorted(set(split_rows[left_split])):
                for right in sorted(set(split_rows[right_split])):
                    pairs.append({"unit": unit, "left": left, "right": right})
    return pairs


def contains_gold_metadata(row: dict) -> bool:
    gold = normalize(re.sub(r"[^\w\s]", " ", row["gold_text"]))
    if len(gold) < 8:
        return False
    metadata = " ".join(
        [row["misconception"], *row["units"]] + [node for edge in row["prerequisites"] for node in edge]
    )
    normalized_metadata = normalize(re.sub(r"[^\w\s]", " ", metadata))
    return gold in normalized_metadata


def build(rows: list[dict], unmatched: dict[str, int]) -> tuple[dict, dict, list[dict]]:
    node_counts = {"train": Counter(), "eval": Counter()}
    edge_counts: Counter = Counter()
    subjects = Counter()
    for row in rows:
        subjects[(row["split"], row["subject"])] += 1
        nodes = set(row["units"])
        for source, target in row["prerequisites"]:
            nodes.update((source, target))
            edge_counts[(source, target)] += 1
        node_counts[row["split"]].update(nodes)

    nodes = set(node_counts["train"]) | set(node_counts["eval"])
    components = connected_components(nodes, edge_counts)
    train_pairs = exact_transfer_pairs(rows, "train", "train")
    train_eval_pairs = exact_transfer_pairs(rows, "train", "eval")
    shared_nodes = set(node_counts["train"]) & set(node_counts["eval"])
    multi_prerequisite_items = sum(len({target for _, target in row["prerequisites"]}) >= 2 for row in rows)
    gold_overlap = sum(contains_gold_metadata(row) for row in rows)

    graph = {
        "nodes": [
            {"id": node, "train_items": node_counts["train"][node], "eval_items": node_counts["eval"][node]}
            for node in sorted(nodes)
        ],
        "edges": [
            {"source": source, "target": target, "items": count}
            for (source, target), count in sorted(edge_counts.items())
        ],
    }
    summary = {
        "items": {split: sum(row["split"] == split for row in rows) for split in ("train", "eval")},
        "subjects": {f"{split}:{subject}": count for (split, subject), count in sorted(subjects.items())},
        "unmatched": unmatched,
        "nodes": len(nodes),
        "edges": len(edge_counts),
        "connected_components": len(components),
        "largest_component": len(components[0]) if components else 0,
        "train_nodes_repeated": sum(count >= 2 for count in node_counts["train"].values()),
        "train_eval_shared_nodes": len(shared_nodes),
        "train_exact_transfer_pairs": len(train_pairs),
        "train_eval_exact_transfer_pairs": len(train_eval_pairs),
        "multi_prerequisite_items": multi_prerequisite_items,
        "metadata_gold_overlap_items": gold_overlap,
        "top_train_nodes": node_counts["train"].most_common(20),
        "shared_nodes": sorted(shared_nodes),
    }
    transfer_pairs = [{**pair, "pair_type": "train_train"} for pair in train_pairs] + [
        {**pair, "pair_type": "train_eval"} for pair in train_eval_pairs
    ]
    return graph, summary, transfer_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-items", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--eval-items", type=Path, required=True)
    parser.add_argument("--eval-annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    train_rows, train_unmatched = join_items(args.train_items, args.train_annotations, "train")
    eval_rows, eval_unmatched = join_items(args.eval_items, args.eval_annotations, "eval")
    rows = train_rows + eval_rows
    graph, summary, transfer_pairs = build(rows, {"train": train_unmatched, "eval": eval_unmatched})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mastery_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "item_skill_map.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.out_dir / "transfer_pairs.jsonl").write_text("".join(json.dumps(pair) + "\n" for pair in transfer_pairs))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
