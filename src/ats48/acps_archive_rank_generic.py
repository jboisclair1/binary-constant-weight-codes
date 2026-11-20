#!/usr/bin/env python3
"""Format-compatible adapter for the preserved ACPS archive ranker."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
PRESERVED_RANKER = (
    HERE / "ATS_4_5_ACPS" / "reproduction_package" / "source" / "acps_archive_rank.py"
)


def load_preserved_ranker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acps_preserved_archive_rank", PRESERVED_RANKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preserved ACPS ranker: {PRESERVED_RANKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CORE = load_preserved_ranker()


def parse_code(path: Path, n: int, w: int) -> list[int]:
    """Read ATS support, binary, and `$BASE=16` hexadecimal code files."""
    records: list[tuple[str, str]] = []
    base = 0
    encoded_n: int | None = None
    zero_based = False
    for source in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = source.split("#", 1)[0].strip().rstrip("\\").strip()
        if not line:
            continue
        compact = "".join(line.split())
        upper = compact.upper()
        if upper.startswith("$BASE="):
            base = int(compact[6:])
            if base not in (2, 16):
                raise ValueError(f"unsupported $BASE={base} in {path}")
            continue
        if upper.startswith("$N="):
            encoded_n = int(compact[3:])
            if not 1 <= encoded_n <= n:
                raise ValueError(f"invalid $N={encoded_n} for n={n} in {path}")
            continue
        if compact.startswith("$"):
            raise ValueError(f"unsupported ATS directive in {path}: {line}")
        records.append((line, compact))
        if base != 16 and not (len(compact) == n and set(compact) <= {"0", "1"}):
            zero_based |= 0 in [int(value) for value in re.findall(r"-?\d+", line)]

    if base == 16:
        if encoded_n is not None:
            digits_per_word = (encoded_n + 3) // 4
            packed = "".join(compact for _line, compact in records)
            if not packed or len(packed) % digits_per_word or not all(
                character in "0123456789abcdefABCDEF" for character in packed
            ):
                raise ValueError(f"packed hexadecimal codewords are malformed in {path}")
            encoded = [
                packed[offset:offset + digits_per_word]
                for offset in range(0, len(packed), digits_per_word)
            ]
        else:
            encoded = [compact for _line, compact in records]
        words = []
        for token in encoded:
            if not token or len(token) > 16 or not all(
                character in "0123456789abcdefABCDEF" for character in token
            ):
                raise ValueError(f"invalid hexadecimal codeword in {path}: {token}")
            word = int(token, 16)
            if word >> n or word.bit_count() != w:
                raise ValueError(f"invalid hexadecimal weight/range in {path}: {token}")
            words.append(word)
        return words

    words: list[int] = []
    for line, compact in records:
        if len(compact) == n and set(compact) <= {"0", "1"}:
            word = sum((character == "1") << index for index, character in enumerate(compact))
        else:
            coordinates = [int(value) for value in re.findall(r"-?\d+", line)]
            if not coordinates:
                continue
            positions = [value if zero_based else value - 1 for value in coordinates]
            if any(position < 0 or position >= n for position in positions):
                raise ValueError(f"coordinate out of range in {path}: {line}")
            word = sum(1 << position for position in positions)
        if word.bit_count() != w:
            raise ValueError(f"wrong weight in {path}: {line}")
        words.append(word)
    return words


CORE.parse_code = parse_code
for exported_name in dir(CORE):
    if not exported_name.startswith("__") and exported_name not in {"main", "parse_code"}:
        globals().setdefault(exported_name, getattr(CORE, exported_name))


def main() -> int:
    CORE.parse_code = parse_code
    return CORE.main()


if __name__ == "__main__":
    raise SystemExit(main())
