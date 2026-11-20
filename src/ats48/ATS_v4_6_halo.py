#!/usr/bin/env python3
"""ATS 4.6 HALO: early defect-1 promotion and structurally ranked reattacks."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import sys
import time
from pathlib import Path

import ATS_v4_5_halo as v45


HERE = Path(__file__).resolve().parent
CORE = HERE / "ATS_v4_6_cuda.exe"
CORE_EXTRA_ARGS: list[str] = []
CORE_CUDA_MODE = "auto"
NEAR_COLLECTION_MULTIPLIER = 1
NEAR_STATE_RANKER = None
SLICE_GENERIC_REPAIR = False
REPAIR_RESERVE_FRACTION = 0.0
DIRECT_REPAIR_MAX_SWAP = 0
NOVELTY = HERE / "ATS_v4_6_halo_novelty.exe"
HALO = HERE / "ATS_v4_6_halo_expand.exe"


def novelty(path: Path) -> int:
    match = re.search(r"(?:nov_|_n)(\d+)(?:_|\.)", path.name)
    return int(match.group(1)) if match else -1


def halo_rank(path: Path) -> tuple[int, int, int]:
    """Prefer deeper coordinated escapes, then less displaced siblings.

    The final component keeps ordering deterministic for equal structural scores.
    """
    match = re.search(r"_s(\d+)_n(\d+)_(\d+)\.txt$", path.name)
    if not match:
        return (-1, -1, -1)
    swap_radius, sibling_novelty, emission_id = map(int, match.groups())
    return (swap_radius, -sibling_novelty, -emission_id)


def stable_reattack_seed(path: Path, base_seed: int, fallback: int) -> int:
    match = re.search(r"_(\d+)\.txt$", path.name)
    identity = int(match.group(1)) if match else fallback
    return base_seed + 1_000_000 + identity


def publish_child(child: Path, args: argparse.Namespace, parent_size: int) -> Path:
    target_size = parent_size + 1
    words = v45.verify_code(child, args.n, args.d, args.w, target_size)
    destination = args.output_dir / (
        f"A_{args.n}_{args.d}_{args.w}_{target_size}_ATS_4_6_HALO_FINAL.txt")
    shutil.copy2(child, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    elapsed = time.monotonic() - args.started
    print(
        f"HALO_VERIFIED_CHILD size={len(words)} child_hash={digest} "
        f"elapsed_seconds={elapsed:.6f} file={destination}", flush=True,
    )
    completion_tag = "HALO_STAGE_DONE" if args.orchestration_stage else "ATS_V4_6_HALO_DONE"
    print(
        f"{completion_tag} base={parent_size} best={len(words)} gain=1 "
        f"retained_siblings=1 elapsed_seconds={elapsed:.6f}", flush=True,
    )
    return destination


def core_attack(
    seed: Path,
    args: argparse.Namespace,
    output: Path,
    seconds: float,
    rng_seed: int,
    deadline: float,
) -> tuple[Path | None, list[Path]]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(CORE), str(seed), str(args.n), str(args.d), str(args.w),
        str(max(0.1, seconds)), str(args.threads), str(output), str(rng_seed),
        "1", "1", "0", "1", "0", str(args.outsider_samples),
        str(args.candidate_pool), CORE_CUDA_MODE,
    ] + CORE_EXTRA_ARGS
    v45.stream_command(command, deadline)
    target_size = len(v45.parse_code(seed)) + 1
    solved = v45.candidates(output, target_size, args.n, args.d, args.w, 0)
    near = v45.candidates(output, target_size, args.n, args.d, args.w, 1)
    return (solved[0] if solved else None), near


def run_search(args: argparse.Namespace) -> int:
    if not 1 <= args.n <= 64:
        raise RuntimeError("unsupported family: require 1 <= n <= 64")
    if not 1 <= args.w < args.n:
        raise RuntimeError("unsupported family: require 1 <= w < n")
    if args.d <= 0 or args.d % 2 or args.d > 2 * args.w:
        raise RuntimeError("unsupported family: require an even distance with 0 < d <= 2w")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise RuntimeError("search time must be a positive finite number")
    if not math.isfinite(args.novelty_seconds) or args.novelty_seconds <= 0:
        raise RuntimeError("novelty time must be a positive finite number")
    if not math.isfinite(args.scout_seconds) or args.scout_seconds <= 0:
        raise RuntimeError("scout time must be a positive finite number")
    if args.threads < 0:
        raise RuntimeError("threads must be nonnegative (0 selects automatic sizing)")
    if args.outsider_samples <= 0 or args.candidate_pool <= 0:
        raise RuntimeError("outsider samples and candidate pool must be positive")
    if args.novelty_steps_per_second <= 0 or args.archive_every <= 0:
        raise RuntimeError("novelty rate and archive interval must be positive")
    if args.max_siblings < 0 or args.promoted_defect1_states < 0 or args.max_halo_siblings < 0:
        raise RuntimeError("sibling and promoted-state counts must be nonnegative")
    if not 0 <= args.rng_seed <= (1 << 64) - 1:
        raise RuntimeError("RNG seed must fit an unsigned 64-bit integer")

    parent = v45.verify_code(args.seed, args.n, args.d, args.w)
    parent_size = len(parent)
    target_size = parent_size + 1
    for executable in (CORE, NOVELTY, HALO):
        if not executable.exists():
            raise RuntimeError(f"required ATS 4.6 HALO component not found: {executable}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.started = time.monotonic()
    deadline = args.started + args.seconds

    novelty_budget = min(args.novelty_seconds, max(2.0, args.seconds * 0.10))
    novelty_deadline = min(deadline, args.started + novelty_budget)
    novelty_steps = max(100_000, int(novelty_budget * args.novelty_steps_per_second))
    novelty_dir = args.output_dir / "novelty"
    print(
        f"HALO_PHASE name=novelty parent_size={parent_size} target_size={target_size} "
        f"budget_seconds={novelty_budget:.3f}",
        flush=True,
    )
    code = v45.stream_command([
        str(NOVELTY), str(args.seed), str(novelty_dir), str(args.rng_seed),
        str(novelty_steps), str(args.archive_every), str(args.n), str(args.d), str(args.w),
    ], novelty_deadline)
    direct = v45.candidates(novelty_dir, target_size, args.n, args.d, args.w, 0)
    if direct:
        publish_child(direct[0], args, parent_size)
        return 0
    if code not in (0, 1) and time.monotonic() < novelty_deadline:
        raise RuntimeError(f"novelty helper failed with exit code {code}")

    siblings = sorted(
        v45.candidates(novelty_dir, parent_size, args.n, args.d, args.w, 0),
        key=novelty,
        reverse=True,
    )[:args.max_siblings]
    if args.seed not in siblings:
        siblings.append(args.seed)

    print(
        f"HALO_PHASE name=early_defect1_promotion siblings={len(siblings)} "
        f"stop_after_states={args.promoted_defect1_states}", flush=True,
    )
    near_state_map: dict[tuple[int, ...], Path] = {}
    collection_target = max(
        args.promoted_defect1_states,
        args.promoted_defect1_states * max(1, NEAR_COLLECTION_MULTIPLIER),
    )
    repair_reserve = max(0.0, args.seconds * REPAIR_RESERVE_FRACTION)
    scout_deadline = max(time.monotonic(), deadline - repair_reserve)
    scouts_used = 0
    cycle_empty_scouts = bool(getattr(args, "cycle_empty_scouts", False))
    while siblings:
        now = time.monotonic()
        if now >= deadline or len(near_state_map) >= collection_target:
            break
        if near_state_map and now >= scout_deadline:
            break
        if scouts_used >= len(siblings) and (not cycle_empty_scouts or near_state_map):
            break
        index = scouts_used % len(siblings)
        sibling = siblings[index]
        scout_dir = (
            args.output_dir / "scouts" / f"attempt-{scouts_used:04d}-sibling-{index:04d}"
            if cycle_empty_scouts
            else args.output_dir / "scouts" / f"sibling-{index:04d}"
        )
        active_scout_deadline = scout_deadline if near_state_map else deadline
        child, near = core_attack(
            sibling, args, scout_dir,
            min(args.scout_seconds, max(0.1, active_scout_deadline - time.monotonic())),
            (args.rng_seed + 10_000 + scouts_used) & ((1 << 64) - 1), active_scout_deadline,
        )
        scouts_used += 1
        if child:
            publish_child(child, args, parent_size)
            return 0
        for state in near:
            key = tuple(sorted(v45.parse_code(state)))
            near_state_map.setdefault(key, state)
    near_states = list(near_state_map.values())
    if NEAR_STATE_RANKER is not None:
        near_states.sort(key=lambda state: NEAR_STATE_RANKER(state, args))
    near_states = near_states[:args.promoted_defect1_states]
    print(
        f"HALO_PROMOTION scouts_used={scouts_used} defect1_states={len(near_states)} "
        f"unique_collected={len(near_state_map)} collection_target={collection_target} "
        f"repair_reserve_seconds={repair_reserve:.3f} "
        f"elapsed_seconds={time.monotonic()-args.started:.6f}", flush=True,
    )

    if DIRECT_REPAIR_MAX_SWAP > 0:
        for index, near in enumerate(near_states):
            if time.monotonic() >= deadline:
                break
            direct_dir = args.output_dir / "direct_repair" / f"near-{index:04d}"
            code = v45.stream_command([
                str(HALO), str(near), str(direct_dir), str(DIRECT_REPAIR_MAX_SWAP),
                str(args.seed), str(args.n), str(args.d), str(args.w), "1",
            ], deadline)
            direct = v45.candidates(direct_dir, target_size, args.n, args.d, args.w, 0)
            print(
                f"HALO_DIRECT_REPAIR state={near.name} max_swap={DIRECT_REPAIR_MAX_SWAP} "
                f"status={'verified' if direct else 'exhausted'}", flush=True,
            )
            if direct:
                publish_child(direct[0], args, parent_size)
                return 0
            if code not in (0, 1) and time.monotonic() < deadline:
                raise RuntimeError(f"direct HALO repair failed with exit code {code}")

    for index, near in enumerate(near_states):
        if time.monotonic() >= deadline:
            break
        repair_deadline = deadline
        if SLICE_GENERIC_REPAIR:
            remaining = deadline - time.monotonic()
            repair_deadline = time.monotonic() + max(
                0.1, remaining / max(1, len(near_states) - index))
        repaired, radius, examined, status = v45.generic_defect1_repair(
            near,
            args.output_dir / "generic_repair" / f"near-{index:04d}-verified-child.txt",
            args.n, args.d, args.w,
            deadline=repair_deadline,
        )
        print(
            f"HALO_GENERIC_REPAIR state={near.name} status={status} radius={radius} "
            f"candidates_examined={examined}", flush=True,
        )
        if repaired:
            publish_child(repaired, args, parent_size)
            return 0

    print(f"HALO_PHASE name=gpu_cpu_multiswap_halo defect1_states={len(near_states)}", flush=True)
    halo_siblings: list[Path] = []
    for index, near in enumerate(near_states):
        if time.monotonic() >= deadline:
            break
        halo_dir = args.output_dir / "halo" / f"near-{index:04d}"
        code = v45.stream_command([
            str(HALO), str(near), str(halo_dir), "3", str(args.seed),
            str(args.n), str(args.d), str(args.w),
        ], deadline)
        direct = v45.candidates(halo_dir, target_size, args.n, args.d, args.w, 0)
        if direct:
            publish_child(direct[0], args, parent_size)
            return 0
        if code not in (0, 1, 3):
            raise RuntimeError(f"HALO helper failed with exit code {code}")
        halo_siblings.extend(
            v45.candidates(halo_dir, parent_size, args.n, args.d, args.w, 0))

    ranked = sorted(halo_siblings, key=halo_rank, reverse=True)[:args.max_halo_siblings]
    print(
        "HALO_PHASE name=structural_portfolio "
        f"siblings={len(ranked)} order="
        + ",".join(path.name for path in ranked),
        flush=True,
    )
    for index, sibling in enumerate(ranked):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        slice_seconds = max(0.1, remaining / max(1, len(ranked) - index))
        child, _near = core_attack(
            sibling, args, args.output_dir / "reattacks" / f"rank-{index:04d}",
            slice_seconds, stable_reattack_seed(sibling, args.rng_seed, index), deadline,
        )
        if child:
            publish_child(child, args, parent_size)
            return 0

    elapsed = time.monotonic() - args.started
    completion_tag = "HALO_STAGE_DONE" if args.orchestration_stage else "ATS_V4_6_HALO_DONE"
    print(
        f"{completion_tag} base={parent_size} best={parent_size} gain=0 "
        f"retained_siblings=0 "
        f"elapsed_seconds={elapsed:.6f}", flush=True,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=53043420260825)
    parser.add_argument("--outsider-samples", type=int, default=5_000_000)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--novelty-steps-per-second", type=int, default=100_000)
    parser.add_argument("--novelty-seconds", type=float, default=30.0)
    parser.add_argument("--archive-every", type=int, default=5)
    parser.add_argument("--max-siblings", type=int, default=24)
    parser.add_argument("--scout-seconds", type=float, default=10.0)
    parser.add_argument("--promoted-defect1-states", type=int, default=1)
    parser.add_argument("--max-halo-siblings", type=int, default=84)
    parser.add_argument("--orchestration-stage", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.seed = args.seed.resolve()
        args.output_dir = args.output_dir.resolve()
        return run_search(args)
    except Exception as exc:
        print(f"HALO_ERROR {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
