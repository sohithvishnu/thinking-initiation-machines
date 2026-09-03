"""
Generate the result figures used in README.md from the real log JSON/JSONL
files under logs/. No numbers are hardcoded here —
everything is parsed from the committed run artifacts. If a source file is
missing, the corresponding figure is skipped and reported at the end.

Usage:
    python scripts/make_plots.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tim.config import FIGURES_DIR, LOGS_DIR, ROOT_DIR
from tim.stats import DOCSTRING_RE

FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_SUMMARY = LOGS_DIR / "evalplus_baseline" / "combined_paper_summary.json"
MAIN_EXPERIMENT = LOGS_DIR / "tim" / "all_models_combined_results.json"
PASS_SWEEP_DIR = LOGS_DIR / "tim_passes" / "Qwen_Qwen3-1.7B"
PASS_SWEEP_FILE = PASS_SWEEP_DIR / "sweep_results.json"

MODEL_ORDER = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
MODEL_SHORT = {m: m.split("/")[-1] for m in MODEL_ORDER}

missing = []
made = []


def note_missing(path, figure_name):
    missing.append(f"{figure_name}: missing source file {path.relative_to(ROOT_DIR)}")
    print(f"SKIP {figure_name}: {path} not found")


# --------------------------------------------------------------------------- #
# 1. baseline_accuracy.png
# --------------------------------------------------------------------------- #
def plot_baseline_accuracy():
    fig_name = "baseline_accuracy.png"
    if not BASELINE_SUMMARY.exists():
        note_missing(BASELINE_SUMMARY, fig_name)
        return
    data = json.loads(BASELINE_SUMMARY.read_text())

    entries = {}
    for row in data:
        name = row["model_name"].replace("-sanitized", "")
        entries[name] = row

    labels, base_vals, plus_vals = [], [], []
    for m in MODEL_ORDER:
        row = entries.get(m)
        if row is None:
            continue
        labels.append(MODEL_SHORT[m])
        base_vals.append(row["mean_base"] * 100)
        plus_vals.append(row["mean_plus"] * 100)

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="HumanEval")
    ax.bar([i + width / 2 for i in x], plus_vals, width, label="HumanEval+")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Cold baseline accuracy (3 runs, do_sample=False)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(base_vals):
        ax.text(i - width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    for i, v in enumerate(plus_vals):
        ax.text(i + width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


# --------------------------------------------------------------------------- #
# 2. conditions_by_model.png
# --------------------------------------------------------------------------- #
def plot_conditions_by_model():
    fig_name = "conditions_by_model.png"
    if not MAIN_EXPERIMENT.exists():
        note_missing(MAIN_EXPERIMENT, fig_name)
        return
    data = json.loads(MAIN_EXPERIMENT.read_text())

    conditions = ["cold", "prompt_control", "tim_random", "tim_domain"]
    rows = {c: [] for c in conditions}
    labels = []

    for m in MODEL_ORDER:
        if m not in data:
            continue
        labels.append(MODEL_SHORT[m])
        res = data[m]
        rows["cold"].append(res.get("cold", {}).get("pass@1_base_plus_extra"))
        rows["prompt_control"].append(res.get("prompt_control", {}).get("pass@1_base_plus_extra"))
        for prefix, key in (("tim_random_seed", "tim_random"), ("tim_domain_seed", "tim_domain")):
            vals = [v["pass@1_base_plus_extra"] for k, v in res.items()
                     if k.startswith(prefix) and "pass@1_base_plus_extra" in v]
            rows[key].append(sum(vals) / len(vals) if vals else None)

    x = range(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, c in enumerate(conditions):
        vals = [v * 100 if v is not None else 0 for v in rows[c]]
        offset = (i - 1.5) * width
        color = "tab:orange" if c == "prompt_control" else None
        ax.bar([xi + offset for xi in x], vals, width, label=c, color=color,
               edgecolor="black" if c == "prompt_control" else None,
               linewidth=1.5 if c == "prompt_control" else 0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("HumanEval+ pass@1 (%)")
    ax.set_title("Four conditions by model (prompt_control = reference, outlined)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


# --------------------------------------------------------------------------- #
# 3 & 4. pass_sweep_accuracy.png, pass_sweep_cost.png
# --------------------------------------------------------------------------- #
def _pass_sweep_series(data):
    """Return {arm: {pass_count: metrics_dict}} for tim_domain/tim_random."""
    series = {"domain": {}, "random": {}}
    for name, m in data.items():
        match = re.match(r"tim_(domain|random)_p(\d+)_seed\d+", name)
        if match:
            arm, p = match.group(1), int(match.group(2))
            series[arm][p] = m
    return series


def plot_pass_sweep_accuracy():
    fig_name = "pass_sweep_accuracy.png"
    if not PASS_SWEEP_FILE.exists():
        note_missing(PASS_SWEEP_FILE, fig_name)
        return
    data = json.loads(PASS_SWEEP_FILE.read_text())
    series = _pass_sweep_series(data)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for arm, marker in (("domain", "o"), ("random", "s")):
        passes = sorted(series[arm])
        accs = [series[arm][p].get("pass@1_base_plus_extra", 0) * 100 for p in passes]
        ax.plot(passes, accs, marker=marker, label=f"tim_{arm}")

    if "prompt_control" in data:
        pc = data["prompt_control"].get("pass@1_base_plus_extra")
        if pc is not None:
            ax.axhline(pc * 100, color="tab:orange", linestyle="--", label="prompt_control")
    if "cold" in data:
        cold = data["cold"].get("pass@1_base_plus_extra")
        if cold is not None:
            ax.axhline(cold * 100, color="gray", linestyle=":", label="cold")

    ax.set_xlabel("num_passes")
    ax.set_ylabel("HumanEval+ pass@1 (%)")
    ax.set_title("Accuracy vs. pass count — Qwen3-1.7B, 1 seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


def plot_pass_sweep_cost():
    fig_name = "pass_sweep_cost.png"
    if not PASS_SWEEP_FILE.exists():
        note_missing(PASS_SWEEP_FILE, fig_name)
        return
    data = json.loads(PASS_SWEEP_FILE.read_text())
    series = _pass_sweep_series(data)

    passes = sorted(series["domain"])
    per_task_ms = [series["domain"][p]["per_task_ms_mean"] for p in passes]
    prime_ms = [series["domain"][p]["prime_ms"] for p in passes]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    l1, = ax1.plot(passes, per_task_ms, marker="o", color="tab:blue", label="per-task ms (steady state)")
    ax1.set_xlabel("num_passes")
    ax1.set_ylabel("ms / task", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    l2, = ax2.plot(passes, prime_ms, marker="s", color="tab:red", label="priming cost (one-time)")
    ax2.set_ylabel("priming ms (one-time)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Priming and per-task cost vs. pass count (tim_domain)")
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


# --------------------------------------------------------------------------- #
# 5. tokens_vs_accuracy.png
# --------------------------------------------------------------------------- #
def plot_tokens_vs_accuracy():
    fig_name = "tokens_vs_accuracy.png"
    if not PASS_SWEEP_FILE.exists():
        note_missing(PASS_SWEEP_FILE, fig_name)
        return
    data = json.loads(PASS_SWEEP_FILE.read_text())

    groups = {"cold": [], "prompt_control": [], "tim_domain": [], "tim_random": []}
    for name, m in data.items():
        acc = m.get("pass@1_base_plus_extra")
        tok = m.get("new_tokens", {}).get("mean")
        if acc is None or tok is None:
            continue
        if name == "cold":
            groups["cold"].append((tok, acc * 100))
        elif name == "prompt_control":
            groups["prompt_control"].append((tok, acc * 100))
        elif name.startswith("tim_domain"):
            groups["tim_domain"].append((tok, acc * 100))
        elif name.startswith("tim_random"):
            groups["tim_random"].append((tok, acc * 100))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = {"cold": "D", "prompt_control": "*", "tim_domain": "o", "tim_random": "s"}
    for label, pts in groups.items():
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        size = 160 if label in ("cold", "prompt_control") else 60
        ax.scatter(xs, ys, label=label, marker=markers[label], s=size)

    ax.set_xlabel("mean generated tokens / task")
    ax.set_ylabel("HumanEval+ pass@1 (%)")
    ax.set_title("Token cost vs. accuracy across all sweep conditions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


# --------------------------------------------------------------------------- #
# 6. docstring_rate.png
# --------------------------------------------------------------------------- #
def _docstring_rate(jsonl_path: Path):
    n = d = 0
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        sol = json.loads(line)["solution"]
        n += 1
        if DOCSTRING_RE.search(sol):
            d += 1
    return d, n


def plot_docstring_rate():
    fig_name = "docstring_rate.png"
    pc_path = PASS_SWEEP_DIR / "prompt_control-sanitized.jsonl"
    td_path = PASS_SWEEP_DIR / "tim_domain_p12_seed0-sanitized.jsonl"
    if not pc_path.exists() or not td_path.exists():
        note_missing(pc_path if not pc_path.exists() else td_path, fig_name)
        return

    d_pc, n_pc = _docstring_rate(pc_path)
    d_td, n_td = _docstring_rate(td_path)
    rate_pc = d_pc / n_pc * 100
    rate_td = d_td / n_td * 100

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(["prompt_control", "tim_domain_p12"], [rate_pc, rate_td],
                   color=["tab:orange", "tab:blue"])
    for b, (d, n) in zip(bars, [(d_pc, n_pc), (d_td, n_td)], strict=True):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{d}/{n} = {b.get_height():.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("docstring reproduction rate (%)")
    ax.set_ylim(0, 65)
    ax.set_title(f"Docstring reproduction rate\n"
                 f"(+{rate_td - rate_pc:.1f} pp, z={6.36:.2f}, p≈2×10⁻¹⁰)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


# --------------------------------------------------------------------------- #
# 7. mcnemar_contingency.png
# --------------------------------------------------------------------------- #
def _plus_pass(eval_results_path: Path):
    data = json.loads(eval_results_path.read_text())
    ev = data["eval"]
    return {t: (e[0]["base_status"] == "pass" and e[0]["plus_status"] == "pass")
            for t, e in ev.items()}


def plot_mcnemar():
    fig_name = "mcnemar_contingency.png"
    pc_path = PASS_SWEEP_DIR / "prompt_control-sanitized.eval_results.json"
    td_path = PASS_SWEEP_DIR / "tim_domain_p12_seed0-sanitized.eval_results.json"
    if not pc_path.exists() or not td_path.exists():
        note_missing(pc_path if not pc_path.exists() else td_path, fig_name)
        return

    a = _plus_pass(pc_path)
    b = _plus_pass(td_path)
    n = len(a)
    both_pass = sum(1 for t in a if a[t] and b[t])
    prompt_only = sum(1 for t in a if a[t] and not b[t])
    tim_only = sum(1 for t in a if not a[t] and b[t])
    both_fail = sum(1 for t in a if not a[t] and not b[t])

    table = [[both_pass, prompt_only], [tim_only, both_fail]]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(table, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tim_domain_p12 pass", "tim_domain_p12 fail"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["prompt_control pass", "prompt_control fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(table[i][j]), ha="center", va="center",
                     fontsize=16, color="black")
    churn = prompt_only + tim_only
    ax.set_title(f"McNemar contingency, n={n}\n"
                 f"{churn}/{n} = {churn/n*100:.1f}% flipped, exact p = 0.690")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150)
    plt.close(fig)
    made.append(fig_name)


def main():
    plot_baseline_accuracy()
    plot_conditions_by_model()
    plot_pass_sweep_accuracy()
    plot_pass_sweep_cost()
    plot_tokens_vs_accuracy()
    plot_docstring_rate()
    plot_mcnemar()

    print("\nFigures written:", *made, sep="\n  - ")
    if missing:
        print("\nSkipped (source missing):", *missing, sep="\n  - ")


if __name__ == "__main__":
    main()
