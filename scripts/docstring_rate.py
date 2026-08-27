"""
Standalone docstring-reproduction audit.

Counts, for every *-sanitized.jsonl file under the repo's logs directories,
the fraction of solutions that contain a docstring immediately after the
`def` line — i.e. the model echoed the problem's docstring back into its
completion rather than just writing code. Prints a table sorted by rate,
highest first.

Usage:
    python scripts/docstring_rate.py [root_dir]

    root_dir defaults to the repository root (parent of this scripts/ dir).
"""

import json
import re
import sys
from pathlib import Path

DOCSTRING_RE = re.compile(r'def\s+\w+\([^)]*\)[^:]*:\s*\n\s*("""|\'\'\')')

EXCLUDE_DIR_NAMES = {"models", ".idea", "__pycache__"}


def docstring_rate(path: Path):
    n = d = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            sol = json.loads(line)["solution"]
        except (json.JSONDecodeError, KeyError):
            continue
        n += 1
        if DOCSTRING_RE.search(sol):
            d += 1
    return d, n


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    files = [
        p for p in sorted(root.rglob("*-sanitized.jsonl"))
        if not EXCLUDE_DIR_NAMES & set(p.parts)
    ]

    if not files:
        print(f"No *-sanitized.jsonl files found under {root}")
        return

    rows = []
    for path in files:
        d, n = docstring_rate(path)
        if n == 0:
            continue
        rows.append((d / n, d, n, path.relative_to(root)))

    rows.sort(reverse=True)

    print(f"{'rate':>7}  {'count':>9}  path")
    print("-" * 90)
    for rate, d, n, rel in rows:
        print(f"{rate*100:6.1f}%  {d:>4}/{n:<4}  {rel}")


if __name__ == "__main__":
    main()
