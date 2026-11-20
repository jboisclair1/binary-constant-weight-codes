#!/usr/bin/env python3
"""Rank correlated ATS near-code endpoints as ACPS pre-seeds.

The script searches an ATS output tree for target-size ``*.best`` near-codes.
A near-code whose conflict graph has a small vertex cover is repaired by deleting
that cover and refilling the resulting vacancies with compatible parent words.
The restored same-size siblings are ranked by exchange displacement, cap defect,
and Hall-nucleus structure relative to the supplied base code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


def parse_code(path: Path, n: int, w: int) -> list[int]:
    lines: list[str] = []
    zero_based = False
    for source in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = source.split("#", 1)[0].strip().rstrip("\\").strip()
        if not line:
            continue
        lines.append(line)
        compact = "".join(line.split())
        if not (len(compact) == n and set(compact) <= {"0", "1"}):
            zero_based |= 0 in [int(value) for value in re.findall(r"-?\d+", line)]

    words: list[int] = []
    for line in lines:
        compact = "".join(line.split())
        if len(compact) == n and set(compact) <= {"0", "1"}:
            word = sum((character == "1") << index for index, character in enumerate(compact))
        else:
            coordinates = [int(value) for value in re.findall(r"-?\d+", line)]
            if not coordinates:
                continue
            word = sum(1 << (value if zero_based else value - 1) for value in coordinates)
        if word.bit_count() != w:
            raise ValueError(f"wrong weight in {path}: {line}")
        words.append(word)
    return words


def compatible(first: int, second: int, lambda_value: int) -> bool:
    return (first & second).bit_count() <= lambda_value


def conflicts(words: list[int], lambda_value: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for index, first in enumerate(words):
        for previous, second in enumerate(words[:index]):
            if not compatible(first, second, lambda_value):
                result.append((previous, index))
    return result


def verify(words: list[int], w: int, lambda_value: int) -> bool:
    return (
        len(words) == len(set(words))
        and all(word.bit_count() == w for word in words)
        and not conflicts(words, lambda_value)
    )


def write_code(path: Path, words: list[int], n: int, d: int, w: int, note: str) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write(f"# {len(words)} constant-weight codewords; {note}\n")
        output.write(f"# A({n},{d},{w}) >= {len(words)}; one-based supports\n\n")
        for word in sorted(words):
            support = ",".join(str(index + 1) for index in range(n) if word >> index & 1)
            output.write("{" + support + "}\\\n")


def maximum_matching(adjacency: list[list[int]], right_count: int) -> int:
    match_right = [-1] * right_count

    def augment(left: int, seen: list[bool]) -> bool:
        for right in adjacency[left]:
            if seen[right]:
                continue
            seen[right] = True
            if match_right[right] < 0 or augment(match_right[right], seen):
                match_right[right] = left
                return True
        return False

    matching = 0
    for left in range(len(adjacency)):
        matching += augment(left, [False] * right_count)
    return matching


def component_summary(adjacency: list[list[int]], right_count: int) -> list[dict[str, int]]:
    reverse: list[list[int]] = [[] for _ in range(right_count)]
    for left, neighbors in enumerate(adjacency):
        for right in neighbors:
            reverse[right].append(left)
    seen_left = [False] * len(adjacency)
    seen_right = [False] * right_count
    components: list[dict[str, int]] = []
    for root in range(len(adjacency)):
        if seen_left[root]:
            continue
        queue: deque[tuple[int, int]] = deque([(0, root)])
        seen_left[root] = True
        left_count = right_vertices = edge_count = 0
        while queue:
            side, vertex = queue.popleft()
            if side == 0:
                left_count += 1
                edge_count += len(adjacency[vertex])
                for right in adjacency[vertex]:
                    if not seen_right[right]:
                        seen_right[right] = True
                        queue.append((1, right))
            else:
                right_vertices += 1
                for left in reverse[vertex]:
                    if not seen_left[left]:
                        seen_left[left] = True
                        queue.append((0, left))
        components.append(
            {
                "left": left_count,
                "right": right_vertices,
                "edges": edge_count,
                "surplus": left_count - right_vertices,
            }
        )
    return sorted(components, key=lambda item: (-item["surplus"], -(item["left"] + item["right"])))


@dataclass
class Endpoint:
    source: str
    removed_indices: list[int]
    cover_size: int
    filler_count: int
    words: list[int]
    displacement: int
    matching: int
    edge_count: int
    blocker_degree_histogram: dict[int, int]
    components: list[dict[str, int]]
    cap_defect: int
    sha256: str


def analyze_endpoint(
    base: list[int],
    endpoint: list[int],
    removed_caps: list[int],
    source: Path,
    removed_indices: list[int],
    filler_count: int,
    lambda_value: int,
) -> Endpoint:
    base_set = set(base)
    endpoint_set = set(endpoint)
    deletions = sorted(base_set - endpoint_set)
    additions = sorted(endpoint_set - base_set)
    deletion_index = {word: index for index, word in enumerate(deletions)}
    adjacency: list[list[int]] = []
    right_degree = [0] * len(deletions)
    edge_count = 0
    for addition in additions:
        neighbors = [
            deletion_index[word]
            for word in deletions
            if not compatible(addition, word, lambda_value)
        ]
        adjacency.append(neighbors)
        edge_count += len(neighbors)
        for right in neighbors:
            right_degree[right] += 1
    matching = maximum_matching(adjacency, len(deletions)) if additions else 0
    cap_defect = min(
        (sum(not compatible(cap, word, lambda_value) for word in endpoint)
         for cap in removed_caps if cap not in endpoint_set),
        default=999,
    )
    canonical = "\n".join(str(word) for word in sorted(endpoint)).encode("ascii")
    return Endpoint(
        source=str(source),
        removed_indices=removed_indices,
        cover_size=len(removed_indices),
        filler_count=filler_count,
        words=endpoint,
        displacement=len(deletions),
        matching=matching,
        edge_count=edge_count,
        blocker_degree_histogram=dict(sorted(Counter(degree for degree in right_degree if degree).items())),
        components=component_summary(adjacency, len(deletions)) if additions else [],
        cap_defect=cap_defect,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )



def enumerate_minimum_vertex_covers(
    edges: list[tuple[int, int]], max_size: int = 3, limit: int = 128
) -> list[tuple[int, ...]]:
    """Enumerate minimum vertex covers of a small conflict graph by edge branching."""
    if not edges:
        return [tuple()]

    def search(
        remaining: tuple[tuple[int, int], ...],
        selected: frozenset[int],
        target: int,
        output: set[tuple[int, ...]],
    ) -> None:
        if len(output) >= limit or len(selected) > target:
            return
        if not remaining:
            if len(selected) == target:
                output.add(tuple(sorted(selected)))
            return
        if len(selected) == target:
            return
        first, second = remaining[0]
        for vertex in (first, second):
            next_edges = tuple(edge for edge in remaining if vertex not in edge)
            search(next_edges, selected | {vertex}, target, output)

    edge_tuple = tuple((first, second) for first, second in edges)
    for size in range(1, max_size + 1):
        covers: set[tuple[int, ...]] = set()
        search(edge_tuple, frozenset(), size, covers)
        if covers:
            return sorted(covers)[:limit]
    return []


def restored_siblings_from_cover(
    base: list[int],
    near: list[int],
    cover: tuple[int, ...],
    w: int,
    lambda_value: int,
    limit: int = 128,
) -> list[tuple[list[int], list[int]]]:
    """Delete a conflict cover and refill to the base cardinality with parent words."""
    core = [word for index, word in enumerate(near) if index not in set(cover)]
    if not verify(core, w, lambda_value):
        return []
    vacancies = len(base) - len(core)
    if vacancies < 0:
        return []
    if vacancies == 0:
        return [(core, [])]

    core_set = set(core)
    eligible = [
        word
        for word in base
        if word not in core_set
        and all(compatible(word, existing, lambda_value) for existing in core)
    ]
    if len(eligible) < vacancies:
        return []

    # Parent words are mutually compatible, so only compatibility with the repaired
    # core is required. Enumerate a bounded, deterministic set of filler choices.
    from itertools import combinations

    restored: list[tuple[list[int], list[int]]] = []
    for fillers in combinations(eligible, vacancies):
        sibling = core + list(fillers)
        if verify(sibling, w, lambda_value):
            restored.append((sibling, list(fillers)))
            if len(restored) >= limit:
                break
    return restored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    lambda_value = args.w - args.d // 2
    base = parse_code(args.base, args.n, args.w)
    if not verify(base, args.w, lambda_value):
        raise ValueError("base code is invalid")

    direct_gains: list[dict[str, object]] = []
    endpoints: list[Endpoint] = []
    seen_endpoint_hashes: set[str] = set()
    sources = sorted(path for path in args.archive.rglob("*.best") if path.is_file())
    for source in sources:
        near = parse_code(source, args.n, args.w)
        if len(near) != len(base) + 1:
            continue
        bad_edges = conflicts(near, lambda_value)
        if not bad_edges and verify(near, args.w, lambda_value):
            canonical = "\n".join(str(word) for word in sorted(near)).encode("ascii")
            direct_gains.append(
                {
                    "source": str(source),
                    "size": len(near),
                    "sha256": hashlib.sha256(canonical).hexdigest(),
                }
            )
            write_code(
                args.output / f"direct_gain_{len(direct_gains):03d}.txt",
                near,
                args.n,
                args.d,
                args.w,
                "ACPS trajectory archive direct gain",
            )
            continue

        covers = enumerate_minimum_vertex_covers(bad_edges, max_size=3, limit=128)
        for cover in covers:
            removed_caps = [near[index] for index in cover]
            for endpoint_words, fillers in restored_siblings_from_cover(
                base, near, cover, args.w, lambda_value, limit=128
            ):
                endpoint = analyze_endpoint(
                    base,
                    endpoint_words,
                    removed_caps,
                    source,
                    list(cover),
                    len(fillers),
                    lambda_value,
                )
                if endpoint.sha256 in seen_endpoint_hashes:
                    continue
                seen_endpoint_hashes.add(endpoint.sha256)
                endpoints.append(endpoint)

    endpoints.sort(
        key=lambda endpoint: (
            endpoint.cap_defect,
            -endpoint.displacement,
            -min(endpoint.blocker_degree_histogram, default=0),
            -endpoint.edge_count,
        )
    )

    rows: list[dict[str, object]] = []
    for rank, endpoint in enumerate(endpoints, start=1):
        filename = f"endpoint_{rank:03d}_disp{endpoint.displacement}_cap{endpoint.cap_defect}.txt"
        write_code(
            args.output / filename,
            endpoint.words,
            args.n,
            args.d,
            args.w,
            "ACPS ranked trajectory endpoint pre-seed",
        )
        rows.append(
            {
                "rank": rank,
                "file": filename,
                "source": endpoint.source,
                "removed_indices": endpoint.removed_indices,
                "cover_size": endpoint.cover_size,
                "filler_count": endpoint.filler_count,
                "displacement": endpoint.displacement,
                "matching": endpoint.matching,
                "edge_count": endpoint.edge_count,
                "cap_defect": endpoint.cap_defect,
                "blocker_degree_histogram": endpoint.blocker_degree_histogram,
                "components": endpoint.components,
                "sha256": endpoint.sha256,
            }
        )

    (args.output / "archive_results.json").write_text(
        json.dumps(
            {
                "base_size": len(base),
                "sources_scanned": len(sources),
                "direct_gains": direct_gains,
                "ranked_endpoints": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output / "archive_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "rank",
                "file",
                "source",
                "removed_indices",
                "cover_size",
                "filler_count",
                "displacement",
                "matching",
                "edge_count",
                "cap_defect",
                "sha256",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    print(
        json.dumps(
            {
                "sources_scanned": len(sources),
                "direct_gains": len(direct_gains),
                "ranked_endpoints": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0 if direct_gains else 2


if __name__ == "__main__":
    raise SystemExit(main())
