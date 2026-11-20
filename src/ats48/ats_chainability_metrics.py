#!/usr/bin/env python3
"""Capture ATS-reached basin metrics for chainability experiments.

This module is observational: telemetry failures must never change a search
result.  It can also backfill the same schema from archived 4.8 level folders.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FAMILY_RE = re.compile(r"A_(\d+)_(\d+)_(\d+)")
LEVEL_RE = re.compile(r"level-(\d+)-to-(\d+)$")
CHECKPOINT_RECORD_RE = re.compile(
    r"^(\d+) fingerprint=(\S+) probe_min=([0-9.]+) probe_near=(\d+) "
    r"degree_cost=(\d+) lineage=(.*?) file=(.*)$"
)


def parse_code(path: Path, n: int, w: int) -> list[int]:
    blocks: list[int] = []
    zero_based = False
    for raw in path.read_text(encoding="utf-8-sig", errors="strict").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "zero-based" in raw.lower():
            zero_based = True
        if not line or line.startswith("$"):
            continue
        if re.fullmatch(r"[01]+", line) and len(line) == n:
            mask = sum(1 << index for index, value in enumerate(line) if value == "1")
        elif "{" in line and "}" in line:
            values = [int(value) for value in re.findall(r"\d+", line)]
            offset = 0 if zero_based or 0 in values else 1
            mask = sum(1 << (value - offset) for value in values)
        else:
            continue
        if mask.bit_count() != w or mask >> n:
            raise ValueError(f"invalid codeword in {path}: {raw}")
        blocks.append(mask)
    if not blocks:
        raise ValueError(f"no codewords parsed from {path}")
    if len(set(blocks)) != len(blocks):
        raise ValueError(f"duplicate codeword in {path}")
    return blocks


def coefficient_of_variation(values: list[int]) -> float:
    mean = statistics.fmean(values) if values else 0.0
    return statistics.pstdev(values) / mean if mean else 0.0


def structural_metrics(blocks: list[int], n: int, d: int, w: int) -> dict[str, Any]:
    q = w - d // 2 + 1
    if d % 2 or q < 1 or q > w:
        raise ValueError(f"invalid family A({n},{d},{w})")
    coordinates = [0] * n
    pairs = [0] * math.comb(n, 2)
    pair_index = {
        pair: index for index, pair in enumerate(itertools.combinations(range(n), 2))
    }
    for block in blocks:
        support = [index for index in range(n) if (block >> index) & 1]
        for coordinate in support:
            coordinates[coordinate] += 1
        for pair in itertools.combinations(support, 2):
            pairs[pair_index[pair]] += 1
    occupancy = len(blocks) * math.comb(w, q) / math.comb(n, q)
    return {
        "q": q,
        "parent_size": len(blocks),
        "shadow_occupancy": occupancy,
        "shadow_slack": 1.0 - occupancy,
        "coordinate_cv": coefficient_of_variation(coordinates),
        "coordinate_min": min(coordinates),
        "coordinate_max": max(coordinates),
        "pair_cv": coefficient_of_variation(pairs),
        "pair_min": min(pairs),
        "pair_max": max(pairs),
    }


def normalized_entropy(frequencies: Counter[str]) -> float:
    total = sum(frequencies.values())
    if total <= 1 or len(frequencies) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in frequencies.values()
    )
    return entropy / math.log(len(frequencies))


def checkpoint_metrics(level_dir: Path) -> dict[str, Any]:
    files = list(level_dir.rglob("ATS*_checkpoint.txt"))
    records: list[dict[str, Any]] = []
    for path in files:
        checkpoint_level = None
        checkpoint_size = None
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            level_match = re.match(r"^## level (\d+) size=(\d+)$", line)
            if level_match:
                checkpoint_level, checkpoint_size = map(int, level_match.groups())
                continue
            record_match = CHECKPOINT_RECORD_RE.match(line)
            if not record_match:
                continue
            index, fingerprint, probe_min, probe_near, degree_cost, lineage, source = record_match.groups()
            records.append(
                {
                    "checkpoint": str(path),
                    "level": checkpoint_level,
                    "size": checkpoint_size,
                    "index": int(index),
                    "fingerprint": fingerprint,
                    "probe_min": float(probe_min),
                    "probe_near": int(probe_near),
                    "degree_cost": int(degree_cost),
                    "lineage": lineage,
                    "source": source,
                }
            )
    probe_values = [float(record["probe_min"]) for record in records]
    near_values = [int(record["probe_near"]) for record in records]
    parent_records = [record for record in records if record["level"] == 0]
    return {
        "checkpoint_files": len(files),
        "checkpoint_records": len(records),
        "checkpoint_parent_records": len(parent_records),
        "checkpoint_unique_states": len({record["fingerprint"] for record in records}),
        "checkpoint_probe_min": min(probe_values) if probe_values else None,
        "checkpoint_probe_mean": statistics.fmean(probe_values) if probe_values else None,
        "checkpoint_probe_near_total": sum(near_values),
        "checkpoint_probe_near_max": max(near_values) if near_values else 0,
        "checkpoint_parent_probe_min": (
            min(float(record["probe_min"]) for record in parent_records)
            if parent_records
            else None
        ),
        "checkpoint_parent_probe_near": sum(int(record["probe_near"]) for record in parent_records),
    }


def low_defect_metrics(level_dir: Path) -> dict[str, Any]:
    files = list(level_dir.rglob("*.low_defect.tsv"))
    defects: Counter[int] = Counter()
    signatures: Counter[str] = Counter()
    state_hashes: set[str] = set()
    parent_hashes: set[str] = set()
    actors: set[str] = set()
    sources: Counter[str] = Counter()
    move_classes: Counter[str] = Counter()
    rows = 0
    for path in files:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                try:
                    defect = int(row.get("defect", ""))
                except ValueError:
                    continue
                rows += 1
                defects[defect] += 1
                if row.get("conflict_signature"):
                    signatures[row["conflict_signature"]] += 1
                if row.get("state_hash"):
                    state_hashes.add(row["state_hash"])
                if row.get("parent_hash"):
                    parent_hashes.add(row["parent_hash"])
                if row.get("actor_id"):
                    actors.add(row["actor_id"])
                if row.get("source"):
                    sources[row["source"]] += 1
                if row.get("move_class"):
                    move_classes[row["move_class"]] += 1
    reservoir = sum(
        weight * defects[defect]
        for defect, weight in ((1, 1.0), (2, 0.3), (3, 0.08), (4, 0.02))
    )
    return {
        "low_defect_files": len(files),
        "low_defect_rows": rows,
        "defect_histogram": {str(key): defects[key] for key in sorted(defects)},
        "accessible_reservoir": reservoir,
        "low_defect_unique_states": len(state_hashes),
        "low_defect_unique_parents": len(parent_hashes),
        "low_defect_unique_signatures": len(signatures),
        "low_defect_signature_entropy": normalized_entropy(signatures),
        "low_defect_unique_actors": len(actors),
        "low_defect_sources": dict(sorted(sources.items())),
        "low_defect_move_classes": dict(sorted(move_classes.items())),
    }


def acps_metrics(level_dir: Path) -> dict[str, Any]:
    files = list(level_dir.rglob("archive_results.json"))
    endpoints: list[dict[str, Any]] = []
    direct_gains = 0
    sources_scanned = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        direct_gains += len(payload.get("direct_gains", []))
        sources_scanned += int(payload.get("sources_scanned", 0))
        endpoints.extend(payload.get("ranked_endpoints", []))
    residuals = [
        max(0, int(endpoint.get("displacement", 0)) - int(endpoint.get("matching", 0)))
        for endpoint in endpoints
    ]
    cap_defects = [int(endpoint.get("cap_defect", 999)) for endpoint in endpoints]
    displacements = [int(endpoint.get("displacement", 0)) for endpoint in endpoints]
    covers = [int(endpoint.get("cover_size", 0)) for endpoint in endpoints]
    return {
        "acps_archive_files": len(files),
        "acps_sources_scanned": sources_scanned,
        "acps_direct_gains": direct_gains,
        "acps_endpoints": len(endpoints),
        "acps_zero_matching_residual": sum(value == 0 for value in residuals),
        "acps_min_matching_residual": min(residuals) if residuals else None,
        "acps_min_cap_defect": min(cap_defects) if cap_defects else None,
        "acps_min_cover_size": min(covers) if covers else None,
        "acps_min_displacement": min(displacements) if displacements else None,
        "acps_max_displacement": max(displacements) if displacements else None,
    }


def regeneration_metrics(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    if previous is None:
        return {
            "reservoir_regeneration": None,
            "probe_near_regeneration": None,
        }
    previous_reservoir = float(previous.get("accessible_reservoir", 0.0))
    current_reservoir = float(current.get("accessible_reservoir", 0.0))
    previous_near = int(previous.get("checkpoint_parent_probe_near", 0))
    current_near = int(current.get("checkpoint_parent_probe_near", 0))
    return {
        "reservoir_regeneration": (
            current_reservoir / previous_reservoir if previous_reservoir > 0 else None
        ),
        "probe_near_regeneration": current_near / previous_near if previous_near > 0 else None,
    }


def collect_level_metrics(
    level_dir: Path,
    n: int,
    d: int,
    w: int,
    parent_size: int,
    target_size: int,
    success: bool,
    result_code: int,
    elapsed_seconds: float | None = None,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent_path = level_dir / "inputs" / "normalized_parent.txt"
    blocks = parse_code(parent_path, n, w)
    if len(blocks) != parent_size:
        raise ValueError(f"parent has {len(blocks)} words; expected {parent_size}")
    metrics: dict[str, Any] = {
        "schema": "ats-chainability-v1",
        "level_dir": str(level_dir.resolve()),
        "n": n,
        "d": d,
        "w": w,
        "parent_size": parent_size,
        "target_size": target_size,
        "success": bool(success),
        "gain": 1 if success else 0,
        "result_code": int(result_code),
        "elapsed_seconds": elapsed_seconds,
        **structural_metrics(blocks, n, d, w),
        **checkpoint_metrics(level_dir),
        **low_defect_metrics(level_dir),
        **acps_metrics(level_dir),
    }
    metrics.update(regeneration_metrics(metrics, previous))
    return metrics


def write_level_metrics(
    metrics: dict[str, Any], level_dir: Path, chain_jsonl: Path | None = None
) -> None:
    destination = level_dir / "chainability_metrics.json"
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    if chain_jsonl is not None:
        with chain_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metrics, separators=(",", ":")) + "\n")


def successful_level(level_dir: Path, n: int, d: int, w: int, target: int) -> bool:
    pattern = re.compile(rf"A_{n}_{d}_{w}_{target}.*(?:FINAL|CHAIN)\.txt$", re.IGNORECASE)
    return any(pattern.match(path.name) for path in level_dir.rglob("*.txt"))


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def backfill(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    records: list[dict[str, Any]] = []
    grouped: dict[Path, list[tuple[int, int, int, int, int, Path]]] = {}
    for parent_path in root.glob("**/level-*-to-*/inputs/normalized_parent.txt"):
        level_dir = parent_path.parent.parent
        family_match = FAMILY_RE.fullmatch(level_dir.parent.name)
        level_match = LEVEL_RE.fullmatch(level_dir.name)
        if family_match is None or level_match is None:
            continue
        n, d, w = map(int, family_match.groups())
        parent_size, target_size = map(int, level_match.groups())
        grouped.setdefault(level_dir.parent, []).append(
            (parent_size, target_size, n, d, w, level_dir)
        )

    for chain_root, levels in grouped.items():
        previous = None
        for parent_size, target_size, n, d, w, level_dir in sorted(levels):
            success = successful_level(level_dir, n, d, w, target_size)
            try:
                metrics = collect_level_metrics(
                    level_dir,
                    n,
                    d,
                    w,
                    parent_size,
                    target_size,
                    success,
                    0 if success else 2,
                    previous=previous,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                print(f"SKIP {level_dir}: {exc}")
                continue
            metrics["chain_root"] = str(chain_root.resolve())
            records.append(metrics)
            previous = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    csv_path = args.output.with_suffix(".csv")
    if records:
        fields = list(records[0])
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flatten(record) for record in records)
    print(
        json.dumps(
            {
                "schema": "ats-chainability-v1",
                "chains": len(grouped),
                "levels": len(records),
                "jsonl": str(args.output),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backfill_parser = subparsers.add_parser("backfill")
    backfill_parser.add_argument("--root", type=Path, required=True)
    backfill_parser.add_argument("--output", type=Path, required=True)
    backfill_parser.set_defaults(func=backfill)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
