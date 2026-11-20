#!/usr/bin/env python3
"""ATS 4.6 HALO + ACPS: generic cover-ranked HALO archive pre-seeding."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

import ATS_v4_5_acps as acps


HERE = Path(__file__).resolve().parent
HALO_RUNNER = HERE / "ATS_v4_6_halo.py"
DONE_TAG = "ATS_V4_6_HALO_ACPS_DONE"


def halo_command(
    seed: Path,
    output: Path,
    seconds: float,
    rng_seed: int,
    args: argparse.Namespace,
    max_halo_siblings: int | None = None,
) -> list[str]:
    return [
        sys.executable,
        str(HALO_RUNNER),
        "--seed", str(seed),
        "--n", str(args.n),
        "--d", str(args.d),
        "--w", str(args.w),
        "--seconds", f"{max(0.1, seconds):.6f}",
        "--threads", str(args.threads),
        "--output-dir", str(output),
        "--rng-seed", str(rng_seed),
        "--outsider-samples", str(args.outsider_samples),
        "--candidate-pool", str(args.candidate_pool),
        "--novelty-seconds", str(args.novelty_seconds),
        "--max-siblings", str(args.max_siblings),
        "--scout-seconds", str(args.scout_seconds),
        "--promoted-defect1-states", str(args.promoted_defect1_states),
        "--max-halo-siblings", str(
            args.max_halo_siblings if max_halo_siblings is None else max_halo_siblings
        ),
        "--orchestration-stage",
    ]


def publish_child(
    source: Path,
    args: argparse.Namespace,
    base_size: int,
    started: float,
    mode: str,
) -> Path:
    words = acps.require_code(source, args.n, args.d, args.w, base_size + 1)
    destination = args.output_dir / (
        f"A_{args.n}_{args.d}_{args.w}_{base_size + 1}_ATS_4_6_HALO_ACPS_FINAL.txt"
    )
    shutil.copy2(source, destination)
    digest = acps.canonical_set_sha256(words, args.n)
    elapsed = time.monotonic() - started
    print(
        f"HALO_ACPS_VERIFIED_CHILD size={len(words)} canonical_sha256={digest} "
        f"mode={mode} elapsed_seconds={elapsed:.6f} file={destination}",
        flush=True,
    )
    print(
        f"ATS_V4_6_HALO_ACPS_DONE base={base_size} best={len(words)} gain=1 "
        f"mode={mode} elapsed_seconds={elapsed:.6f}",
        flush=True,
    )
    return destination


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.n <= 64:
        raise RuntimeError("unsupported family: require 1 <= n <= 64")
    if not 1 <= args.w < args.n:
        raise RuntimeError("unsupported family: require 1 <= w < n")
    if args.d <= 0 or args.d % 2 or args.d > 2 * args.w:
        raise RuntimeError("unsupported family: require an even distance with 0 < d <= 2w")
    for value, label in (
        (args.seconds, "search time"),
        (args.halo_discovery_seconds, "HALO discovery time"),
        (args.novelty_seconds, "novelty time"),
        (args.scout_seconds, "scout time"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{label} must be a positive finite number")
    if args.threads < 0:
        raise RuntimeError("threads must be nonnegative (0 selects automatic sizing)")
    if args.outsider_samples <= 0 or args.candidate_pool <= 0:
        raise RuntimeError("outsider samples and candidate pool must be positive")
    if args.endpoint_beam <= 0 or args.endpoint_cycles <= 0:
        raise RuntimeError("endpoint beam and cycles must be positive")
    if args.max_siblings < 0:
        raise RuntimeError("maximum polished siblings must be nonnegative")
    if args.promoted_defect1_states <= 0 or args.max_halo_siblings <= 0:
        raise RuntimeError("promoted defect-1 states and HALO siblings must be positive")
    if not 0 <= args.rng_seed <= (1 << 64) - 1:
        raise RuntimeError("RNG seed must fit an unsigned 64-bit integer")


def emit_not_found(base_size: int, started: float, status: str) -> None:
    elapsed = time.monotonic() - started
    print(
        f"{DONE_TAG} base={base_size} best={base_size} gain=0 "
        f"status={status} elapsed_seconds={elapsed:.6f}",
        flush=True,
    )


def run_search(args: argparse.Namespace) -> int:
    validate_args(args)
    if not HALO_RUNNER.exists():
        raise RuntimeError(f"ATS 4.6 HALO runner not found: {HALO_RUNNER}")
    if not acps.RANKER_PATH.exists():
        raise RuntimeError(f"ACPS archive ranker not found: {acps.RANKER_PATH}")

    source_seed = args.seed.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trusted_seed_size = getattr(args, "trusted_seed_size", None)
    if trusted_seed_size is None:
        base = acps.require_code(source_seed, args.n, args.d, args.w)
    else:
        # The 4.8 chain verifies its initial seed and every selected child before
        # handing that child to the next level. Reparse the immutable handoff for
        # normalization, but do not repeat the quadratic validity check.
        base = acps.parse_code(source_seed, args.n, args.w)
        if len(base) != trusted_seed_size:
            raise RuntimeError(
                f"trusted seed {source_seed} has {len(base)} words; "
                f"expected {trusted_seed_size}"
            )
    base_size = len(base)
    target_size = base_size + 1
    normalized_seed = args.output_dir / "inputs" / "normalized_parent.txt"
    normalized_seed.parent.mkdir(parents=True, exist_ok=True)
    acps.RANKER.write_code(
        normalized_seed,
        base,
        args.n,
        args.d,
        args.w,
        "normalized HALO + ACPS parent",
    )
    args.seed = normalized_seed.resolve()
    started = time.monotonic()
    deadline = started + args.seconds
    print(
        f"HALO_ACPS_CONFIG family=A({args.n},{args.d},{args.w}) "
        f"base={base_size} target={target_size} source_seed={source_seed} "
        f"normalized_seed={args.seed} "
        f"rng_seed={args.rng_seed}",
        flush=True,
    )

    refresh = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.5:
            emit_not_found(base_size, started, "TIME_LIMIT_REACHED")
            return 2

        # Keep the original directory layout for the first pass. Subsequent
        # endurance refreshes are isolated so their archives and rank manifests
        # remain reproducible and never overwrite an earlier search basin.
        round_root = (
            args.output_dir if refresh == 0 else
            args.output_dir / "portfolio_refreshes" / f"round-{refresh + 1:03d}"
        )
        round_seed = (args.rng_seed + refresh * 10_000_019) & ((1 << 64) - 1)
        if refresh:
            print(
                f"HALO_ACPS_PORTFOLIO_REFRESH round={refresh + 1} "
                f"remaining_seconds={remaining:.3f} rng_seed={round_seed}",
                flush=True,
            )

        discovery_seconds = min(
            args.halo_discovery_seconds,
            max(0.1, remaining),
            max(0.1, args.seconds * 0.5),
        )
        discovery_dir = round_root / "halo_discovery"
        print(
            f"HALO_ACPS_PHASE name=halo_archive_discovery refresh={refresh + 1} "
            f"base={base_size} target={target_size} "
            f"budget_seconds={discovery_seconds:.3f}",
            flush=True,
        )
        code = acps.stream_command(
            halo_command(
                args.seed,
                discovery_dir,
                discovery_seconds,
                round_seed,
                args,
                max_halo_siblings=0,
            ),
            deadline,
        )
        child = acps.best_valid_candidate(
            discovery_dir, args.n, args.d, args.w, target_size,
        )
        if child is not None:
            publish_child(child, args, base_size, started, "halo_discovery")
            return 0
        if code not in (0, 2) and time.monotonic() < deadline:
            raise RuntimeError(f"HALO discovery failed with exit code {code}")

        if time.monotonic() >= deadline - 0.5:
            emit_not_found(base_size, started, "TIME_LIMIT_REACHED")
            return 2
        rank_dir = round_root / "acps_rank"
        print(
            f"HALO_ACPS_PHASE name=acps_cover_rank refresh={refresh + 1} "
            f"archive={discovery_dir}",
            flush=True,
        )
        rank_code = acps.stream_command(
            [
                sys.executable,
                str(acps.RANKER_PATH),
                str(args.seed),
                str(discovery_dir),
                str(args.n),
                str(args.d),
                str(args.w),
                str(rank_dir),
            ],
            deadline,
        )
        direct = acps.best_valid_candidate(
            rank_dir, args.n, args.d, args.w, target_size,
        )
        if direct is not None:
            publish_child(direct, args, base_size, started, "acps_direct_hall")
            return 0
        if rank_code not in (0, 2) and time.monotonic() < deadline:
            raise RuntimeError(f"ACPS ranker failed with exit code {rank_code}")

        endpoints = acps.load_ranked_endpoints(rank_dir)[:args.endpoint_beam]
        if not endpoints:
            endpoints = [args.seed]
        print(
            f"HALO_ACPS_RANKED refresh={refresh + 1} endpoints={len(endpoints)} order="
            + ",".join(path.name for path in endpoints),
            flush=True,
        )

        endpoint_cycles = max(1, args.endpoint_cycles)
        total_slots = len(endpoints) * endpoint_cycles
        completed_slots = 0
        print(
            f"HALO_ACPS_PARENT_PORTFOLIO refresh={refresh + 1} "
            f"parents={len(endpoints)} cycles={endpoint_cycles} "
            f"reattack_slots={total_slots}",
            flush=True,
        )
        indexed_endpoints = list(enumerate(endpoints))
        for cycle in range(endpoint_cycles):
            # Rotate equal-ranked work so an interrupted long run does not always
            # privilege the same endpoint order on every structural look-back pass.
            offset = (cycle + refresh) % len(indexed_endpoints)
            cycle_endpoints = indexed_endpoints[offset:] + indexed_endpoints[:offset]
            for cycle_index, (rank_index, endpoint) in enumerate(cycle_endpoints):
                remaining = deadline - time.monotonic()
                if remaining <= 0.5:
                    break
                acps.require_code(endpoint, args.n, args.d, args.w, base_size)
                remaining_slots = max(1, total_slots - completed_slots)
                slice_seconds = max(0.1, remaining / remaining_slots)
                if args.runtime_policy == "deadline":
                    # A refresh is a diversification unit. Do not hand one finite
                    # stage the entire endurance budget; refresh discovery/ranking
                    # after every configured group of endpoint cycles.
                    slice_seconds = min(slice_seconds, args.halo_discovery_seconds)
                if endpoint_cycles == 1:
                    output = round_root / "halo_reattacks" / f"endpoint-{rank_index + 1:03d}"
                else:
                    output = (
                        round_root / "halo_reattacks" / f"cycle-{cycle + 1:03d}"
                        / f"endpoint-{rank_index + 1:03d}"
                    )
                print(
                    f"HALO_ACPS_PHASE name=halo_ranked_reattack "
                    f"refresh={refresh + 1} cycle={cycle + 1} "
                    f"endpoint={rank_index + 1} slot={completed_slots + 1}/{total_slots} "
                    f"budget_seconds={slice_seconds:.3f} seed={endpoint}",
                    flush=True,
                )
                code = acps.stream_command(
                    halo_command(
                        endpoint,
                        output,
                        slice_seconds,
                        (
                            round_seed + 2_000_000 +
                            cycle * 1_000_000 + cycle_index
                        ) & ((1 << 64) - 1),
                        args,
                    ),
                    deadline,
                )
                completed_slots += 1
                child = acps.best_valid_candidate(
                    output, args.n, args.d, args.w, target_size,
                )
                if child is not None:
                    publish_child(
                        child, args, base_size, started,
                        "acps_ranked_halo_reattack",
                    )
                    return 0
                if code not in (0, 2) and time.monotonic() < deadline:
                    raise RuntimeError(
                        f"ranked HALO reattack failed with exit code {code}"
                    )
            if time.monotonic() >= deadline - 0.5:
                break

        if time.monotonic() >= deadline - 0.5:
            emit_not_found(base_size, started, "TIME_LIMIT_REACHED")
            return 2
        if args.runtime_policy == "bounded":
            emit_not_found(base_size, started, "NOT_FOUND_BOUNDED")
            return 2
        refresh += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=6208002)
    parser.add_argument("--outsider-samples", type=int, default=5_000_000)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--halo-discovery-seconds", type=float, default=60.0)
    parser.add_argument("--endpoint-beam", type=int, default=2)
    parser.add_argument("--endpoint-cycles", type=int, default=1)
    parser.add_argument(
        "--runtime-policy",
        choices=("bounded", "deadline"),
        default="bounded",
        help=(
            "bounded stops after one discovery/ranking portfolio; deadline "
            "refreshes diversified portfolios until the search time expires"
        ),
    )
    parser.add_argument("--novelty-seconds", type=float, default=30.0)
    parser.add_argument("--max-siblings", type=int, default=24)
    parser.add_argument("--scout-seconds", type=float, default=10.0)
    parser.add_argument("--promoted-defect1-states", type=int, default=1)
    parser.add_argument("--max-halo-siblings", type=int, default=84)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_search(args)
    except Exception as exc:
        print(f"HALO_ACPS_ERROR {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
