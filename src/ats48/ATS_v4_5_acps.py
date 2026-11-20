#!/usr/bin/env python3
"""ATS 4.5 ACPS orchestration and deterministic reproduction for Frontier Lab.

Live mode wraps the existing ATS 4.5 CUDA executable with Alternating-Cover
Pre-Seeding (ACPS): short ATS probes create a correlated target-size archive,
the ACPS ranker repairs small conflict covers into valid same-size siblings, and
ATS reattacks the most displaced low-cap-defect endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
ACPS_HOME = HERE / "ATS_4_5_ACPS"
PACKAGE = ACPS_HOME / "reproduction_package"
RANKER_PATH = HERE / "acps_archive_rank_generic.py"
CUDA_CORE = HERE / "ATS_v4_5_cuda.exe"
sys.dont_write_bytecode = True


def load_ranker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ats45_acps_archive_rank", RANKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ACPS ranker: {RANKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RANKER = load_ranker()


def parse_code(path: Path, n: int, w: int) -> list[int]:
    return RANKER.parse_code(path, n, w)


def verify_words(words: list[int], n: int, d: int, w: int) -> bool:
    lambda_value = w - d // 2
    return (
        0 < n <= 64
        and d > 0
        and d % 2 == 0
        and 0 < w <= n
        and all(word >> n == 0 for word in words)
        and RANKER.verify(words, w, lambda_value)
    )


def require_code(path: Path, n: int, d: int, w: int, expected: int | None = None) -> list[int]:
    words = parse_code(path, n, w)
    if expected is not None and len(words) != expected:
        raise RuntimeError(f"{path} has {len(words)} words; expected {expected}")
    if not verify_words(words, n, d, w):
        raise RuntimeError(f"invalid A({n},{d},{w}) code: {path}")
    return words


def canonical_set_sha256(words: list[int], n: int) -> str:
    payload = "\n".join(f"{word:0{n}b}" for word in sorted(words)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def verify_metrics(words: list[int], n: int, d: int, w: int) -> dict[str, object]:
    if not verify_words(words, n, d, w):
        raise RuntimeError("code failed exact verification")
    minimum_distance = 2 * w if len(words) < 2 else 2 * w
    maximum_intersection = 0
    pair_checks = 0
    for index, first in enumerate(words):
        for second in words[:index]:
            intersection = (first & second).bit_count()
            maximum_intersection = max(maximum_intersection, intersection)
            minimum_distance = min(minimum_distance, 2 * (w - intersection))
            pair_checks += 1
    return {
        "valid": True,
        "n": n,
        "d": d,
        "w": w,
        "count": len(words),
        "distinct": len(set(words)),
        "minimum_distance": minimum_distance,
        "maximum_intersection": maximum_intersection,
        "pair_checks": pair_checks,
        "canonical_set_sha256": canonical_set_sha256(words, n),
    }


def stream_command(command: list[str], deadline: float | None = None) -> int:
    print("> " + subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while True:
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                print(line, end="", flush=True)
            if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                process.terminate()
                deadline = None
        reader.join(timeout=1)
        return process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise


def publish_code(source: Path, args: argparse.Namespace, label: str) -> Path:
    words = require_code(source, args.n, args.d, args.w)
    destination = args.output_dir / f"A_{args.n}_{args.d}_{args.w}_{len(words)}_ATS_4_5_ACPS_{label}.txt"
    RANKER.write_code(
        destination,
        words,
        args.n,
        args.d,
        args.w,
        f"ATS 4.5 ACPS {label.lower()}",
    )
    digest = canonical_set_sha256(words, args.n)
    print(
        f"ACPS_VERIFIED_CHILD size={len(words)} canonical_sha256={digest} file={destination}",
        flush=True,
    )
    print(f"A({args.n},{args.d},{args.w}) >= {len(words)}", flush=True)
    return destination


def candidate_files(root: Path) -> list[Path]:
    return sorted({*root.rglob("*.txt"), *root.rglob("*.best")})


def best_valid_candidate(
    root: Path, n: int, d: int, w: int, minimum_size: int
) -> Path | None:
    best: tuple[int, Path] | None = None
    for path in candidate_files(root):
        try:
            words = parse_code(path, n, w)
        except (OSError, ValueError):
            continue
        if len(words) < minimum_size or not verify_words(words, n, d, w):
            continue
        item = (len(words), path)
        if best is None or item[0] > best[0] or (item[0] == best[0] and str(item[1]) < str(best[1])):
            best = item
    return best[1] if best is not None else None


def verify_backtest(directory: Path) -> dict[str, object]:
    payload = json.loads((directory / "acps_result.json").read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    n, d, w = (int(parameters[key]) for key in ("n", "d", "w"))
    neutral = require_code(directory / "neutral_preseed.txt", n, d, w, int(payload["parent_size"]))
    reconstructed = require_code(
        directory / "reconstructed_gain.txt", n, d, w, int(payload["reconstructed_size"])
    )
    caps = parse_code(directory / "appendable_caps.txt", n, w)
    if len(caps) != int(payload["appendable_caps"]):
        raise RuntimeError(f"cap count mismatch in {directory}")
    if set(neutral) | set(caps) != set(reconstructed) or set(neutral) & set(caps):
        raise RuntimeError(f"neutral-plus-cap reconstruction mismatch in {directory}")
    return {
        "name": directory.name,
        "parent_size": len(neutral),
        "reconstructed_size": len(reconstructed),
        "appendable_caps": len(caps),
        "valid": True,
    }


def run_reproduction(args: argparse.Namespace) -> int:
    if (args.n, args.d, args.w) != (62, 10, 7):
        raise RuntimeError("the bundled deterministic reproduction is A(62,10,7)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frontier = PACKAGE / "new_frontier"
    print("ACPS_PHASE name=reproduction_verify", flush=True)
    parent = require_code(frontier / "A_62_10_7_753_parent.txt", 62, 10, 7, 753)
    neutral = require_code(frontier / "A_62_10_7_753_ACPS_neutral_preseed.txt", 62, 10, 7, 753)
    cap = require_code(frontier / "A_62_10_7_appendable_cap.txt", 62, 10, 7, 1)
    child_path = frontier / "A_62_10_7_754_canonical.txt"
    child = require_code(child_path, 62, 10, 7, 754)

    if set(neutral) | set(cap) != set(child) or set(neutral) & set(cap):
        raise RuntimeError("neutral pre-seed plus cap does not equal the 754-word certificate")
    parent_set, child_set = set(parent), set(child)
    deletions = sorted(parent_set - child_set)
    additions = sorted(child_set - parent_set)
    adjacency = [
        [right for right, incumbent in enumerate(deletions) if not RANKER.compatible(word, incumbent, 2)]
        for word in additions
    ]
    exchange_edges = sum(map(len, adjacency))
    matching = RANKER.maximum_matching(adjacency, len(deletions))
    deficiency = len(additions) - matching
    neutral_displacement = len(set(neutral) - parent_set)
    cap_conflicts = sum(not RANKER.compatible(cap[0], word, 2) for word in neutral)
    metrics = verify_metrics(child, 62, 10, 7)

    expected = {
        "deleted_words": 28,
        "inserted_words": 29,
        "exchange_edges": 92,
        "maximum_matching": 28,
        "hall_deficiency": 1,
        "neutral_displacement": 28,
        "cap_conflicts": 0,
    }
    observed = {
        "deleted_words": len(deletions),
        "inserted_words": len(additions),
        "exchange_edges": exchange_edges,
        "maximum_matching": matching,
        "hall_deficiency": deficiency,
        "neutral_displacement": neutral_displacement,
        "cap_conflicts": cap_conflicts,
    }
    if observed != expected:
        raise RuntimeError(f"A62 ACPS decomposition mismatch: {observed}")
    if metrics["canonical_set_sha256"] != "6c237e7058ff57fc4abdc7df3a12ba2aa860a361f379eafb1e145e21b414bbb5":
        raise RuntimeError("A62 canonical-set SHA-256 mismatch")

    print("ACPS_PHASE name=backtest_verify", flush=True)
    backtests = [verify_backtest(path) for path in sorted((PACKAGE / "backtests").iterdir()) if path.is_dir()]
    report = {
        "algorithm": "ATS 4.5 ACPS",
        "status": "REPRODUCED",
        "certificate": metrics,
        "transition": observed,
        "backtests": backtests,
        "prospective_results": json.loads((PACKAGE / "RESULT_SUMMARY.json").read_text(encoding="utf-8")),
    }
    report_path = args.output_dir / "ATS_4_5_ACPS_REPRODUCTION.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final = publish_code(child_path, args, "REPRO")
    print(
        "ATS_V4_5_ACPS_REPRO_OK "
        f"parent=753 child=754 matching={matching} deficiency={deficiency} "
        f"neutral_displacement={neutral_displacement} caps=1 backtests={len(backtests)} "
        f"report={report_path} final={final}",
        flush=True,
    )
    return 0


def cuda_probe_command(
    parent: Path,
    output: Path,
    seconds: float,
    rng_seed: int,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(CUDA_CORE),
        str(parent),
        str(args.n),
        str(args.d),
        str(args.w),
        f"{seconds:.3f}",
        str(args.threads),
        str(output),
        str(rng_seed),
        "1",
        "1",
        "0",
        "1",
        "0",
        str(args.outsider_samples),
        str(args.candidate_pool),
        "auto",
    ]


def load_ranked_endpoints(rank_directory: Path) -> list[Path]:
    results_path = rank_directory / "archive_results.json"
    if not results_path.exists():
        return []
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    endpoints: list[Path] = []
    for row in payload.get("ranked_endpoints", []):
        path = rank_directory / str(row["file"])
        if path.exists():
            endpoints.append(path)
    return endpoints


def run_live(args: argparse.Namespace) -> int:
    if args.seed is None:
        raise RuntimeError("--seed is required for a live ATS 4.5 ACPS search")
    if not CUDA_CORE.exists():
        raise RuntimeError(f"ATS 4.5 CUDA core not found: {CUDA_CORE}")
    if not RANKER_PATH.exists():
        raise RuntimeError(f"ACPS archive ranker not found: {RANKER_PATH}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.seed = args.seed.resolve()
    base = require_code(args.seed, args.n, args.d, args.w)
    base_size = len(base)
    deadline = time.monotonic() + args.seconds
    archive_root = args.output_dir / "correlated_archive"
    parents = [args.seed]
    best_preseed: Path | None = None
    seen_parent_hashes = {canonical_set_sha256(base, args.n)}
    probe_number = 0

    for cycle in range(args.cycles):
        if time.monotonic() >= deadline - 0.5:
            break
        print(f"ACPS_PHASE name=correlated_archive cycle={cycle + 1}", flush=True)
        current_parents = parents[: max(1, args.archive_probes)]
        for slot in range(args.archive_probes):
            remaining = deadline - time.monotonic()
            if remaining <= 0.5:
                break
            parent = current_parents[slot % len(current_parents)]
            probe_number += 1
            probe_dir = archive_root / f"cycle-{cycle + 1:02d}" / f"probe-{probe_number:04d}"
            probe_dir.mkdir(parents=True, exist_ok=True)
            slots_left = max(1, args.archive_probes - slot + args.endpoint_beam)
            probe_seconds = min(args.scout_seconds, max(0.5, remaining / slots_left))
            code = stream_command(
                cuda_probe_command(parent, probe_dir, probe_seconds, args.rng_seed + probe_number, args),
                deadline,
            )
            if code not in (0, 2) and time.monotonic() < deadline:
                raise RuntimeError(f"ATS 4.5 archive probe failed with exit code {code}")
            gain = best_valid_candidate(probe_dir, args.n, args.d, args.w, base_size + 1)
            if gain is not None:
                final = publish_code(gain, args, "FINAL")
                print(
                    f"ATS_V4_5_ACPS_DONE base={base_size} best={len(parse_code(final, args.n, args.w))} "
                    f"gain={len(parse_code(final, args.n, args.w)) - base_size} mode=ats_archive",
                    flush=True,
                )
                return 0

        if time.monotonic() >= deadline - 0.25:
            break
        print(f"ACPS_PHASE name=cover_rank cycle={cycle + 1}", flush=True)
        rank_dir = args.output_dir / "acps_rank" / f"cycle-{cycle + 1:02d}"
        rank_command = [
            sys.executable,
            str(RANKER_PATH),
            str(args.seed),
            str(archive_root),
            str(args.n),
            str(args.d),
            str(args.w),
            str(rank_dir),
        ]
        rank_code = stream_command(rank_command, deadline)
        if rank_code not in (0, 2) and time.monotonic() < deadline:
            raise RuntimeError(f"ACPS archive ranker failed with exit code {rank_code}")
        direct = best_valid_candidate(rank_dir, args.n, args.d, args.w, base_size + 1)
        if direct is not None:
            publish_code(direct, args, "FINAL")
            print(f"ATS_V4_5_ACPS_DONE base={base_size} best={base_size + 1} gain=1 mode=direct_hall", flush=True)
            return 0

        endpoints = load_ranked_endpoints(rank_dir)[: args.endpoint_beam]
        if endpoints:
            best_preseed = endpoints[0]
        print(f"ACPS_RANKED cycle={cycle + 1} endpoints={len(endpoints)}", flush=True)
        print(f"ACPS_PHASE name=ranked_reattack cycle={cycle + 1}", flush=True)
        next_parents: list[Path] = []
        for endpoint_index, endpoint in enumerate(endpoints):
            remaining = deadline - time.monotonic()
            if remaining <= 0.5:
                break
            endpoint_words = require_code(endpoint, args.n, args.d, args.w, base_size)
            endpoint_hash = canonical_set_sha256(endpoint_words, args.n)
            if endpoint_hash not in seen_parent_hashes:
                seen_parent_hashes.add(endpoint_hash)
                next_parents.append(endpoint)
            reattack_dir = args.output_dir / "reattacks" / f"cycle-{cycle + 1:02d}" / f"endpoint-{endpoint_index + 1:03d}"
            reattack_dir.mkdir(parents=True, exist_ok=True)
            reattack_seconds = max(0.5, remaining / max(1, len(endpoints) - endpoint_index))
            code = stream_command(
                cuda_probe_command(
                    endpoint,
                    reattack_dir,
                    reattack_seconds,
                    args.rng_seed + 1_000_000 + cycle * 10_000 + endpoint_index,
                    args,
                ),
                deadline,
            )
            if code not in (0, 2) and time.monotonic() < deadline:
                raise RuntimeError(f"ATS 4.5 ACPS reattack failed with exit code {code}")
            gain = best_valid_candidate(reattack_dir, args.n, args.d, args.w, base_size + 1)
            if gain is not None:
                final = publish_code(gain, args, "FINAL")
                best_size = len(parse_code(final, args.n, args.w))
                print(
                    f"ATS_V4_5_ACPS_DONE base={base_size} best={best_size} "
                    f"gain={best_size - base_size} mode=ranked_reattack",
                    flush=True,
                )
                return 0
        parents = next_parents or [args.seed]

    if best_preseed is not None:
        published = publish_code(best_preseed, args, "PRESEED")
        print(f"ACPS_RETAINED_PRESEED size={base_size} file={published}", flush=True)
    print(f"ATS_V4_5_ACPS_DONE base={base_size} best={base_size} gain=0 status=NOT_FOUND_BOUNDED", flush=True)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--n", type=int, default=62)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--w", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=6208002)
    parser.add_argument("--outsider-samples", type=int, default=5_000_000)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--archive-probes", type=int, default=4)
    parser.add_argument("--endpoint-beam", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--scout-seconds", type=float, default=10.0)
    parser.add_argument("--verify-reproduction", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.output_dir = args.output_dir.resolve()
        if args.seconds <= 0 or args.scout_seconds <= 0:
            raise RuntimeError("time budgets must be positive")
        if args.archive_probes <= 0 or args.endpoint_beam <= 0 or args.cycles <= 0:
            raise RuntimeError("archive probes, endpoint beam, and cycles must be positive")
        if args.verify_reproduction:
            return run_reproduction(args)
        return run_live(args)
    except Exception as exc:
        print(f"ACPS_ERROR {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
