#!/usr/bin/env python3
"""ATS 4.8 HALO + ACPS with a fresh runtime window after every verified gain."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import ATS_v4_5_acps as acps
import ATS_v4_6_halo_acps as orchestration
import ats_chainability_metrics as chainability


HERE = Path(__file__).resolve().parent
HALO_RUNNER = HERE / "ATS_v4_8_halo_stage.py"
UINT64_MAX = (1 << 64) - 1


def publish_child(
    source: Path,
    args,
    base_size: int,
    started: float,
    mode: str,
) -> Path:
    # run_search only publishes paths returned by best_valid_candidate, which has
    # already completed the exact validity check. Keep the count assertion while
    # avoiding a second quadratic pass over the same child.
    words = acps.parse_code(source, args.n, args.w)
    if len(words) != base_size + 1:
        raise RuntimeError(
            f"verified child {source} has {len(words)} words; expected {base_size + 1}"
        )
    destination = args.output_dir / (
        f"A_{args.n}_{args.d}_{args.w}_{base_size + 1}_ATS_4_8_HALO_ACPS_FINAL.txt"
    )
    shutil.copy2(source, destination)
    digest = acps.canonical_set_sha256(words, args.n)
    elapsed = time.monotonic() - started
    print(
        f"HALO_ACPS_V48_VERIFIED_CHILD size={len(words)} canonical_sha256={digest} "
        f"mode={mode} elapsed_seconds={elapsed:.6f} file={destination}",
        flush=True,
    )
    print(
        f"HALO_ACPS_V48_LEVEL_DONE base={base_size} best={len(words)} gain=1 "
        f"mode={mode} elapsed_seconds={elapsed:.6f}",
        flush=True,
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = orchestration.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--max-chain-gain",
        type=int,
        default=0,
        help="maximum verified gains to chain; 0 means unlimited",
    )
    return parser


def run_chain(args: argparse.Namespace) -> int:
    if args.max_chain_gain < 0:
        raise RuntimeError("--max-chain-gain must be nonnegative (0 means unlimited)")

    source_seed = args.seed.resolve()
    root_output = args.output_dir.resolve()
    root_output.mkdir(parents=True, exist_ok=True)
    initial_words = acps.require_code(source_seed, args.n, args.d, args.w)
    initial_size = len(initial_words)
    current_seed = source_seed
    current_size = initial_size
    chain_started = time.monotonic()
    gains = 0
    previous_metrics = None

    print(
        f"HALO_ACPS_V48_CHAIN_START base={initial_size} "
        f"max_gain={'unlimited' if args.max_chain_gain == 0 else args.max_chain_gain} "
        f"runtime_window_seconds={args.seconds:.6f} reset_after_gain=1 "
        f"seed={source_seed}",
        flush=True,
    )

    while args.max_chain_gain == 0 or gains < args.max_chain_gain:
        level_output = root_output / f"level-{current_size}-to-{current_size + 1}"
        level_args = argparse.Namespace(**vars(args))
        level_args.seed = current_seed
        level_args.output_dir = level_output
        level_args.seconds = args.seconds
        level_args.rng_seed = (args.rng_seed + gains * 1_000_003) & UINT64_MAX
        level_args.trusted_seed_size = current_size

        print(
            f"HALO_ACPS_V48_CHAIN_LEVEL_START level={gains} parent_size={current_size} "
            f"target_size={current_size + 1} runtime_window_seconds={args.seconds:.6f} "
            f"rng_seed={level_args.rng_seed}",
            flush=True,
        )
        level_started = time.monotonic()
        result = orchestration.run_search(level_args)
        level_elapsed = time.monotonic() - level_started
        candidate = level_output / (
            f"A_{args.n}_{args.d}_{args.w}_{current_size + 1}_ATS_4_8_HALO_ACPS_FINAL.txt"
        )
        level_success = result == 0 and candidate.exists()
        try:
            current_metrics = chainability.collect_level_metrics(
                level_output,
                args.n,
                args.d,
                args.w,
                current_size,
                current_size + 1,
                level_success,
                result,
                elapsed_seconds=level_elapsed,
                previous=previous_metrics,
            )
            chainability.write_level_metrics(
                current_metrics,
                level_output,
                root_output / "chainability_metrics.jsonl",
            )
            previous_metrics = current_metrics
            print(
                "HALO_ACPS_V48_CHAINABILITY "
                f"parent_size={current_size} success={int(level_success)} "
                f"probe_min={current_metrics['checkpoint_parent_probe_min']} "
                f"probe_near={current_metrics['checkpoint_parent_probe_near']} "
                f"reservoir={current_metrics['accessible_reservoir']} "
                f"signatures={current_metrics['low_defect_unique_signatures']} "
                f"acps_endpoints={current_metrics['acps_endpoints']} "
                f"reservoir_regeneration={current_metrics['reservoir_regeneration']}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"HALO_ACPS_V48_CHAINABILITY_WARNING parent_size={current_size} error={exc}",
                file=sys.stderr,
                flush=True,
            )
        if result != 0 or not candidate.exists():
            print(
                f"HALO_ACPS_V48_CHAIN_LEVEL_DONE level={gains} "
                f"parent_size={current_size} target_size={current_size + 1} "
                f"gain=0 result={result}",
                flush=True,
            )
            break

        # The candidate selector already performed the exact quadratic validity
        # check before orchestration.run_search could return success. Re-read only
        # for the canonical digest and count; verifying the same artifact again
        # here was the dominant cost on fast successful Level 1 chains.
        child = acps.parse_code(candidate, args.n, args.w)
        if len(child) != current_size + 1:
            raise RuntimeError(
                f"published child {candidate} has {len(child)} words; "
                f"expected {current_size + 1}"
            )
        gains += 1
        current_size += 1
        chained_output = root_output / (
            f"A_{args.n}_{args.d}_{args.w}_{current_size}_ATS_4_8_HALO_ACPS_CHAIN.txt"
        )
        shutil.copy2(candidate, chained_output)
        current_seed = chained_output
        digest = acps.canonical_set_sha256(child, args.n)
        print(
            f"HALO_ACPS_V48_CHAIN_LEVEL_DONE level={gains - 1} "
            f"parent_size={current_size - 1} target_size={current_size} gain=1 "
            f"canonical_sha256={digest} verified_words={len(child)} "
            f"runtime_reset_seconds={args.seconds:.6f} file={chained_output}",
            flush=True,
        )

    elapsed = time.monotonic() - chain_started
    final_path = current_seed
    if gains:
        final_path = root_output / (
            f"A_{args.n}_{args.d}_{args.w}_{current_size}_ATS_4_8_HALO_ACPS_FINAL.txt"
        )
        shutil.copy2(current_seed, final_path)
    print(
        f"ATS_V4_8_HALO_ACPS_DONE base={initial_size} best={current_size} gain={gains} "
        f"gains={gains} mode=verified_chain elapsed_seconds={elapsed:.6f} "
        f"final={final_path}",
        flush=True,
    )
    return 0 if gains else 2


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not HALO_RUNNER.exists():
            raise RuntimeError(f"ATS 4.8 HALO stage not found: {HALO_RUNNER}")
        orchestration.HALO_RUNNER = HALO_RUNNER
        orchestration.publish_child = publish_child
        orchestration.DONE_TAG = "HALO_ACPS_V48_LEVEL_DONE"
        return run_chain(args)
    except Exception as exc:
        print(f"HALO_ACPS_V48_ERROR {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
