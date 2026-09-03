"""
Test statistics and eval-artifact readers shared by every analysis script.

Previously each analysis script reached into a sibling script with
`sys.path.insert` to borrow these (`analyze_int4_mechanism` and
`analyze_mbpp_scale` and `analyze_rung5` all imported from
`analyze_quant_gap`), which made an analysis module implicitly an entrypoint
and a library at once. They live here instead.

Pass criterion, used everywhere in this repo: a task counts as passed only if
`base_status == 'pass' AND plus_status == 'pass'` — i.e. it passes both the
original tests and the EvalPlus extra tests. This matches EvalPlus's own
`pass@1 (base + extra)` metric; `plus_status` alone would overcount.
"""

import json
import math
import re
from pathlib import Path

# A docstring opened on the line immediately after the `def` line — i.e. the
# model echoed the problem's docstring into its completion instead of just
# writing code. The headline stylistic effect of KV priming.
DOCSTRING_RE = re.compile(r'def\s+\w+\([^)]*\)[^:]*:\s*\n\s*("""|\'\'\')')


def wilson_ci(k: int, n: int, z: float = 1.959963985):
    """Wilson score interval for k successes in n trials. Returns (p, lo, hi).

    Preferred over the normal approximation because several cells here are
    small enough that the Wald interval would run outside [0, 1].
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b: int, c: int):
    """Exact McNemar: a binomial test on the discordant pairs. Returns (b, c, p).

    Exact rather than chi-square because the discordant counts here are often
    well under the ~25 the chi-square approximation wants.
    """
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return b, c, 1.0
    return b, c, binomtest(min(b, c), n, 0.5).pvalue


def load_eval_results(path: Path):
    """Read an EvalPlus `*.eval_results.json`.

    Returns {task_id: {'base_pass': bool, 'plus_pass': bool}}, or None if the
    file is absent — callers report the absence rather than treating a missing
    condition as a failing one.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        task_id: {
            "base_pass": entries[0]["base_status"] == "pass",
            "plus_pass": entries[0]["plus_status"] == "pass",
        }
        for task_id, entries in data["eval"].items()
    }


def plus_pass_count(tasks: dict) -> int:
    return sum(1 for t in tasks.values() if t["base_pass"] and t["plus_pass"])


def base_pass_count(tasks: dict) -> int:
    return sum(1 for t in tasks.values() if t["base_pass"])


def docstring_rate(sanitized_jsonl: Path):
    """(n_with_docstring, n_total) over a `*-sanitized.jsonl`, or None."""
    path = Path(sanitized_jsonl)
    if not path.exists():
        return None
    n = d = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            solution = json.loads(line)["solution"]
        except (json.JSONDecodeError, KeyError):
            continue
        n += 1
        if DOCSTRING_RE.search(solution):
            d += 1
    return d, n
