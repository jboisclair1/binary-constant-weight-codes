#!/usr/bin/env python3
"""ATS 4.5 HALO novelty/defect-1/multiswap orchestration and certificate regression."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import math
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORE = HERE / "ATS_v4_5_cuda.exe"
NOVELTY = HERE / "ATS_v4_5_halo_novelty.exe"
HALO = HERE / "ATS_v4_5_halo_expand.exe"
BUNDLE = HERE.parent / "ATS_4_5_A53_434_REPRO_BUNDLE"
KNOWN_CERTIFICATE = BUNDLE / "codes" / "A_53_10_7_434_ATS_4_5.txt"
KNOWN_SHA256 = "9b090bbe480a72cf130708ebf8923493fde09926f95a6bde2f846f30be720d78"


def parse_code(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    directives = {line.upper() for line in stripped if line.startswith("$")}
    if "$BASE=16" in directives:
        encoded_n = next(
            (int(line[3:]) for line in stripped if line.upper().startswith("$N=")),
            None,
        )
        tokens = ["".join(line.split()) for line in stripped if line and not line.startswith("$")]
        if encoded_n is not None:
            digits_per_word = (encoded_n + 3) // 4
            packed = "".join(tokens)
            if not packed or len(packed) % digits_per_word:
                raise ValueError(f"malformed packed hexadecimal code in {path}")
            tokens = [
                packed[offset:offset + digits_per_word]
                for offset in range(0, len(packed), digits_per_word)
            ]
        if any(
            not token or len(token) > 16
            or not all(character in "0123456789abcdefABCDEF" for character in token)
            for token in tokens
        ):
            raise ValueError(f"invalid hexadecimal codeword in {path}")
        return [int(token, 16) for token in tokens]
    words: list[int] = []
    for body in re.findall(r"\{([^{}]+)\}", text):
        mask = 0
        for token in body.split(","):
            mask |= 1 << (int(token.strip()) - 1)
        words.append(mask)
    if not words:
        for line in text.splitlines():
            bits = line.strip()
            if bits and not bits.startswith(("#", "$")) and set(bits) <= {"0", "1"}:
                words.append(sum((bit == "1") << index for index, bit in enumerate(bits)))
    return words


def exact_defect(words: list[int], n: int, d: int, w: int) -> int:
    if len(set(words)) != len(words) or any(word.bit_count() != w or word >> n for word in words):
        return 1 << 60
    maximum_intersection = w - (d + 1) // 2
    return sum(
        (words[i] & words[j]).bit_count() > maximum_intersection
        for i in range(len(words)) for j in range(i)
    )


def repeated_shadow_defect(words: list[int], n: int, d: int, w: int) -> int:
    """Return the ATS objective for every valid critical-shadow size."""
    if len(set(words)) != len(words) or any(word.bit_count() != w or word >> n for word in words):
        return 1 << 60
    critical_order = w - (d + 1) // 2 + 1
    counts: collections.Counter[int] = collections.Counter()
    for word in words:
        coordinates = [bit for bit in range(n) if word & (1 << bit)]
        for subset in itertools.combinations(coordinates, critical_order):
            counts[sum(1 << bit for bit in subset)] += 1
    # ATS scores every colliding owner pair for each critical shadow.  Using
    # C(count, 2), rather than count - 1, also remains exact when three or more
    # words share the same shadow.
    return sum(math.comb(count, 2) for count in counts.values() if count > 1)


def generic_defect1_repair(
    state: Path, output: Path, n: int, d: int, w: int,
    max_universe: int = 5_000_000,
    deadline: float | None = None,
) -> tuple[Path | None, int, int, str]:
    """Repair a true ATS defect-1 state at radius 1 or 2.

    Critical-shadow order two uses exact bitset clique search, so it does not
    scan or materialize the constant-weight universe.  Other shadow orders use
    the intentionally universe-bounded generic fallback.
    """
    words = parse_code(state)
    if repeated_shadow_defect(words, n, d, w) != 1:
        return None, 0, 0, "not_exact_defect1"
    universe_size = math.comb(n, w)
    maximum_intersection = w - (d + 1) // 2
    critical_order = maximum_intersection + 1
    if critical_order != 2 and universe_size > max_universe:
        return None, 0, 0, f"universe_exceeds_cpu_limit:{universe_size}>{max_universe}"
    conflicts = [
        (i, j) for i in range(len(words)) for j in range(i)
        if (words[i] & words[j]).bit_count() > maximum_intersection
    ]
    if len(conflicts) != 1:
        return None, 0, 0, f"conflict_pair_count:{len(conflicts)}"
    left, right = conflicts[0]
    candidates_examined = 0
    search_nodes = 0
    deadline_exhausted = False

    def weight_masks():
        for support in itertools.combinations(range(n), w):
            yield sum(1 << bit for bit in support)

    def shadows(word: int):
        coordinates = [bit for bit in range(n) if word & (1 << bit)]
        for subset in itertools.combinations(coordinates, critical_order):
            yield sum(1 << bit for bit in subset)

    def pair_shadow_outsiders(core: list[int]):
        """Yield every compatible word as a clique in the unused-pair graph."""
        nonlocal candidates_examined, search_nodes, deadline_exhausted
        all_coordinates = (1 << n) - 1
        adjacency = [all_coordinates ^ (1 << bit) for bit in range(n)]
        for word in core:
            coordinates = [bit for bit in range(n) if word & (1 << bit)]
            for first, second in itertools.combinations(coordinates, 2):
                adjacency[first] &= ~(1 << second)
                adjacency[second] &= ~(1 << first)
        occupied = set(core)

        def cliques(candidates: int, needed: int, chosen: int):
            nonlocal candidates_examined, search_nodes, deadline_exhausted
            search_nodes += 1
            if (search_nodes & 4095) == 0 and deadline is not None \
                    and time.monotonic() >= deadline:
                deadline_exhausted = True
                return
            while candidates.bit_count() >= needed and not deadline_exhausted:
                vertex = candidates & -candidates
                candidates ^= vertex
                if needed == 1:
                    candidate = chosen | vertex
                    candidates_examined += 1
                    if candidate not in occupied:
                        yield candidate
                else:
                    index = vertex.bit_length() - 1
                    yield from cliques(
                        candidates & adjacency[index], needed - 1, chosen | vertex)

        yield from cliques(all_coordinates, w, 0)

    def compatible_outsiders(core: list[int]):
        nonlocal candidates_examined, deadline_exhausted
        if critical_order == 2:
            yield from pair_shadow_outsiders(core)
            return
        forbidden = {shadow for word in core for shadow in shadows(word)}
        occupied = set(core)
        for candidate in weight_masks():
            candidates_examined += 1
            if (candidates_examined & 4095) == 0 and deadline is not None \
                    and time.monotonic() >= deadline:
                deadline_exhausted = True
                return
            if candidate not in occupied and all(
                shadow not in forbidden for shadow in shadows(candidate)
            ):
                yield candidate

    def publish(child: list[int], radius: int) -> tuple[Path, int, int, str]:
        if exact_defect(child, n, d, w) != 0:
            raise RuntimeError("generic defect-1 repair failed independent verification")
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# generic ATS critical-shadow defect-1 radius-{radius} repair",
            f"# A({n},{d},{w}) size={len(child)}",
        ]
        for word in sorted(child):
            support = [str(bit + 1) for bit in range(n) if word & (1 << bit)]
            lines.append("{" + ",".join(support) + "}")
        output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        verify_code(output, n, d, w, len(words))
        return output, radius, candidates_examined, "verified"

    for endpoint in (left, right):
        core = [word for index, word in enumerate(words) if index != endpoint]
        candidate = next(compatible_outsiders(core), None)
        if candidate is not None:
            return publish(core + [candidate], 1)
        if deadline_exhausted:
            return None, 0, candidates_examined, "deadline_exhausted"

    core = [word for index, word in enumerate(words) if index not in (left, right)]
    outsiders: list[int] = []
    for first in compatible_outsiders(core):
        for second in outsiders:
            if (first & second).bit_count() <= maximum_intersection:
                return publish(core + [second, first], 2)
        outsiders.append(first)
    if deadline_exhausted:
        return None, 0, candidates_examined, "deadline_exhausted"
    return None, 0, candidates_examined, "exhausted_radius2"


def verify_code(path: Path, n: int, d: int, w: int, expected_size: int | None = None) -> list[int]:
    words = parse_code(path)
    if expected_size is not None and len(words) != expected_size:
        raise RuntimeError(f"{path} has {len(words)} words; expected {expected_size}")
    defect = exact_defect(words, n, d, w)
    if defect != 0:
        raise RuntimeError(f"{path} failed exact verification: defect={defect}")
    return words


def stream_command(command: list[str], deadline: float | None = None) -> int:
    print("> " + subprocess.list2cmdline(command), flush=True)
    process = subprocess.Popen(
        command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        output_closed = False
        while process.poll() is None or not output_closed:
            try:
                line = lines.get(timeout=0.1)
                if line is None:
                    output_closed = True
                else:
                    print(line, end="", flush=True)
            except queue.Empty:
                pass
            if deadline is not None and time.monotonic() >= deadline:
                process.terminate()
                deadline = None
        reader.join(timeout=1)
        return process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise


def candidates(folder: Path, size: int, n: int, d: int, w: int, defect: int) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".txt", ".best", ""):
            continue
        try:
            words = parse_code(path)
        except (OSError, ValueError):
            continue
        if len(words) == size:
            score = repeated_shadow_defect(words, n, d, w)
            if score == defect:
                found.append((score, path))
    return [path for _score, path in found]


def core_attack(
    seed: Path, args: argparse.Namespace, output: Path, seconds: float,
    rng_seed: int, deadline: float,
) -> tuple[Path | None, list[Path]]:
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(CORE), str(seed), str(args.n), str(args.d), str(args.w), str(max(0.1, seconds)),
        str(args.threads), str(output), str(rng_seed), "1", "1", "0", "1", "0",
        str(args.outsider_samples), str(args.candidate_pool), "auto",
    ]
    stream_command(command, deadline)
    solved = candidates(output, len(parse_code(seed)) + 1, args.n, args.d, args.w, 0)
    near = candidates(output, len(parse_code(seed)) + 1, args.n, args.d, args.w, 1)
    return (solved[0] if solved else None), near


def publish_child(child: Path, args: argparse.Namespace, output_name: str) -> Path:
    words = verify_code(child, args.n, args.d, args.w)
    destination = args.output_dir / output_name
    shutil.copy2(child, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(f"HALO_VERIFIED_CHILD size={len(words)} child_hash={digest} file={destination}", flush=True)
    print(f"ATS_V4_5_HALO_DONE base={len(words)-1} best={len(words)} gain=1 retained_siblings=1", flush=True)
    return destination


def run_regression(args: argparse.Namespace) -> int:
    certificate = args.certificate or KNOWN_CERTIFICATE
    words = verify_code(certificate, args.n, args.d, args.w, 434)
    digest = hashlib.sha256(certificate.read_bytes()).hexdigest()
    if digest != KNOWN_SHA256:
        raise RuntimeError(f"certificate SHA-256 mismatch: {digest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = publish_child(certificate, args, "A_53_10_7_434_ATS_4_5_HALO_REPRO.txt")
    print(f"HALO_REGRESSION_OK words={len(words)} sha256={digest} output={output}", flush=True)
    return 0


def run_search(args: argparse.Namespace) -> int:
    if (args.n, args.d, args.w) != (53, 10, 7):
        raise RuntimeError("ATS 4.5 HALO helpers currently target A(53,10,7)")
    parent = verify_code(args.seed, args.n, args.d, args.w, 433)
    for executable in (CORE, NOVELTY, HALO):
        if not executable.exists():
            raise RuntimeError(f"required HALO component not found: {executable}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.seconds
    print(f"HALO_PHASE name=novelty parent_size={len(parent)}", flush=True)
    novelty_dir = args.output_dir / "novelty"
    novelty_budget = min(60.0, max(2.0, args.seconds * 0.15))
    novelty_deadline = min(deadline, started + novelty_budget)
    novelty_steps = max(100_000, int(novelty_budget * args.novelty_steps_per_second))
    code = stream_command([
        str(NOVELTY), str(args.seed), str(novelty_dir), str(args.rng_seed),
        str(novelty_steps), str(args.archive_every),
    ], novelty_deadline)
    direct = candidates(novelty_dir, 434, args.n, args.d, args.w, 0)
    if direct:
        publish_child(direct[0], args, "A_53_10_7_434_ATS_4_5_HALO_FINAL.txt")
        return 0
    if code not in (0, 1) and time.monotonic() < novelty_deadline:
        raise RuntimeError(f"novelty helper failed with exit code {code}")

    siblings = sorted(
        candidates(novelty_dir, 433, args.n, args.d, args.w, 0),
        key=lambda path: int(re.search(r"nov_(\d+)", path.name).group(1)) if re.search(r"nov_(\d+)", path.name) else -1,
        reverse=True,
    )[:args.max_siblings]
    if args.seed not in siblings:
        siblings.append(args.seed)
    print(f"HALO_PHASE name=defect1_scout siblings={len(siblings)}", flush=True)
    near_states: list[Path] = []
    for index, sibling in enumerate(siblings):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        slice_seconds = min(args.scout_seconds, max(0.1, remaining / max(1, len(siblings) - index)))
        child, near = core_attack(
            sibling, args, args.output_dir / "scouts" / f"sibling-{index:04d}",
            slice_seconds, args.rng_seed + 10_000 + index, deadline,
        )
        if child:
            publish_child(child, args, "A_53_10_7_434_ATS_4_5_HALO_FINAL.txt")
            return 0
        near_states.extend(near)

    print(f"HALO_PHASE name=multiswap_halo defect1_states={len(near_states)}", flush=True)
    halo_siblings: list[Path] = []
    for index, near in enumerate(near_states[:args.max_near_states]):
        if time.monotonic() >= deadline:
            break
        halo_dir = args.output_dir / "halo" / f"near-{index:04d}"
        code = stream_command([str(HALO), str(near), str(halo_dir), "3", str(args.seed)], deadline)
        direct = candidates(halo_dir, 434, args.n, args.d, args.w, 0)
        if direct:
            publish_child(direct[0], args, "A_53_10_7_434_ATS_4_5_HALO_FINAL.txt")
            return 0
        if code not in (0, 1, 3):
            raise RuntimeError(f"halo helper failed with exit code {code}")
        halo_siblings.extend(candidates(halo_dir, 433, args.n, args.d, args.w, 0))

    def novelty(path: Path) -> int:
        match = re.search(r"_n(\d+)_", path.name)
        return int(match.group(1)) if match else -1

    ranked = sorted(halo_siblings, key=novelty, reverse=True)[:args.max_halo_siblings]
    print(f"HALO_PHASE name=ranked_reattack siblings={len(ranked)}", flush=True)
    for index, sibling in enumerate(ranked):
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        slice_seconds = max(0.1, remaining / max(1, len(ranked) - index))
        child, _near = core_attack(
            sibling, args, args.output_dir / "reattacks" / f"halo-{index:04d}",
            slice_seconds, args.rng_seed + 1_000_000 + index, deadline,
        )
        if child:
            publish_child(child, args, "A_53_10_7_434_ATS_4_5_HALO_FINAL.txt")
            return 0
    print("ATS_V4_5_HALO_DONE base=433 best=433 gain=0 retained_siblings=0", flush=True)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--n", type=int, default=53)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--w", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rng-seed", type=int, default=53043420260825)
    parser.add_argument("--outsider-samples", type=int, default=5_000_000)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--novelty-steps-per-second", type=int, default=100_000)
    parser.add_argument("--archive-every", type=int, default=5)
    parser.add_argument("--max-siblings", type=int, default=24)
    parser.add_argument("--scout-seconds", type=float, default=10.0)
    parser.add_argument("--max-near-states", type=int, default=32)
    parser.add_argument("--max-halo-siblings", type=int, default=84)
    parser.add_argument("--verify-reproduction", action="store_true")
    parser.add_argument("--certificate", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.output_dir = args.output_dir.resolve()
        if args.seed is not None:
            args.seed = args.seed.resolve()
        if args.certificate is not None:
            args.certificate = args.certificate.resolve()
        if args.verify_reproduction:
            return run_regression(args)
        if args.seed is None:
            raise RuntimeError("--seed is required for a HALO search")
        return run_search(args)
    except Exception as exc:
        print(f"HALO_ERROR {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
