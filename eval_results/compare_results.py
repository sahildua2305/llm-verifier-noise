"""Compare local evaluation results against the base model and clean RL run.

Run from any directory with ``python eval_results/compare_results.py``.
Uniform-auditing runs are discovered and ordered by their per-batch budget.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

METRICS = ("pass_at_1", "pass_at_5", "pass_at_16")
BASE_RESULT = Path(
    "RGYM_decimal_chain_sum/Qwen-Qwen3-1.7B-Base/"
    "decimal_chain_sum_3_6_3_6_3_6_16_0.6.json"
)
CLEAN_RESULT = Path(
    "RGYM_decimal_chain_sum/runs-20261018_101809_dsum-clean-run-checkpoint-60/"
    "decimal_chain_sum_3_6_3_6_3_6_16_0.6.json"
)
SETTING_KEYS = (
    "provider",
    "max_tokens",
    "temperature",
    "top_p",
    "n_samples",
    "reasoning",
    "reasoning_effort",
)


@dataclass
class Result:
    path: Path
    benchmark: str
    run: str
    category: str
    dataset: str
    count: int
    metrics: dict[str, float]
    settings: dict
    questions: tuple[tuple[str, str], ...]


def load_results(root: Path) -> list[Result]:
    results = []
    for path in sorted(root.rglob("*.json")):
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "categories" not in data:
            continue
        benchmark = "/".join(parts[:-2])
        run = parts[-2]
        settings = data.get("metadata", {})
        for category in data["categories"]:
            for dataset in category.get("datasets", []):
                entries = dataset.get("results", [])
                count = dataset.get("total_examples")
                if count != len(entries):
                    raise ValueError(f"{path}: total_examples does not match results")
                metrics = {key: dataset.get(key) for key in METRICS}
                if any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in metrics.values()
                ):
                    raise ValueError(f"{path}: missing or invalid pass@k metrics")
                questions = tuple(
                    sorted(
                        (str(entry["question"]), str(entry["expected_answer"]))
                        for entry in entries
                    )
                )
                results.append(
                    Result(
                        path,
                        benchmark,
                        run,
                        category["name"],
                        dataset["name"],
                        count,
                        metrics,
                        settings,
                        questions,
                    )
                )
    return results


def run_label(row: Result, root: Path, base_path: Path, clean_path: Path) -> str:
    relative_path = row.path.relative_to(root)
    if relative_path == base_path:
        return "Base model"
    if relative_path == clean_path:
        return "Clean RL"
    budget = re.search(r"(?:uniform-audit-b|audit_uniform_b)(\d+)(?:\D|$)", row.run)
    if budget:
        return f"Uniform audit b{int(budget.group(1))}"
    return "Other"


def sort_key(row: Result, root: Path, base_path: Path, clean_path: Path) -> tuple:
    label = run_label(row, root, base_path, clean_path)
    if label == "Base model":
        return (0, 0, row.run)
    if label == "Clean RL":
        return (1, 0, row.run)
    if label.startswith("Uniform audit b"):
        return (2, int(label.removeprefix("Uniform audit b")), row.run)
    return (3, 0, row.run)


def comparison_status(row: Result, baseline: Result) -> tuple[bool, str]:
    changed = [
        key
        for key in SETTING_KEYS
        if row.settings.get(key) != baseline.settings.get(key)
    ]
    if changed == ["reasoning"] and row.settings.get("provider") == "vllm":
        # This repo's VLLMModel sends the same generation request in both modes.
        # Its reasoning flag only parses the returned text.
        changed = []
        parser_note = "vLLM parser flag differs"
    else:
        parser_note = ""
    if changed:
        return False, "settings differ: " + ", ".join(changed)
    if row.questions != baseline.questions:
        return False, "questions differ"
    return True, parser_note or ("baseline" if row is baseline else "matched")


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def delta_cells(row: Result, baseline: Result | None) -> list[str]:
    if baseline is None or not comparison_status(row, baseline)[0]:
        return ["—"] * len(METRICS)
    deltas = []
    for key in METRICS:
        delta = (
            Decimal(str(row.metrics[key])) - Decimal(str(baseline.metrics[key]))
        ) * 100
        deltas.append(f"{delta.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+}")
    return deltas


def print_group(
    rows: list[Result], root: Path, base_path: Path, clean_path: Path
) -> None:
    first = rows[0]
    rows = sorted(rows, key=lambda row: sort_key(row, root, base_path, clean_path))
    base = next((row for row in rows if row.path.relative_to(root) == base_path), None)
    clean = next(
        (row for row in rows if row.path.relative_to(root) == clean_path), None
    )
    print(f"\n{first.benchmark} / {first.category} / {first.dataset}")

    print("\nAbsolute performance")
    absolute = []
    for index, row in enumerate(rows, 1):
        absolute.append(
            [str(index), run_label(row, root, base_path, clean_path), str(row.count)]
            + [f"{row.metrics[key]:.2%}" for key in METRICS]
        )
    print_table(["#", "Run", "N", "Pass@1", "Pass@5", "Pass@16"], absolute)

    print("\nChange from each baseline (percentage points)")
    differences = []
    for index, row in enumerate(rows, 1):
        differences.append(
            [str(index), run_label(row, root, base_path, clean_path)]
            + delta_cells(row, base)
            + delta_cells(row, clean)
        )
    print_table(
        [
            "#",
            "Run",
            "Base @1",
            "Base @5",
            "Base @16",
            "Clean @1",
            "Clean @5",
            "Clean @16",
        ],
        differences,
    )
    if base is None or clean is None:
        missing = [
            name
            for name, value in (("base model", base), ("clean RL", clean))
            if value is None
        ]
        print(f"Missing baseline in this group: {', '.join(missing)}")
    for index, row in enumerate(rows, 1):
        for name, baseline in (("base model", base), ("clean RL", clean)):
            if baseline is not None:
                comparable, reason = comparison_status(row, baseline)
                if not comparable:
                    print(f"  {index} vs {name}: {reason}")
    if (
        base
        and clean
        and comparison_status(base, clean)[1] == "vLLM parser flag differs"
    ):
        print(
            "Note: The base eval used a different vLLM reasoning-parser flag; generation requests match."
        )
    for index, row in enumerate(rows, 1):
        print(f"  {index}: {row.path.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Local eval_results directory (default: alongside this script)",
    )
    parser.add_argument(
        "--base-result",
        type=Path,
        default=BASE_RESULT,
        help="Base-model JSON path relative to --root",
    )
    parser.add_argument(
        "--clean-result",
        type=Path,
        default=CLEAN_RESULT,
        help="Clean-RL JSON path relative to --root",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results = load_results(root)
    if not results:
        parser.error(f"No evaluation result JSON files found under {root}")
    groups = defaultdict(list)
    for result in results:
        groups[(result.benchmark, result.category, result.dataset)].append(result)
    for group in sorted(groups):
        print_group(groups[group], root, args.base_result, args.clean_result)
    print("\nPass@1 estimates one-sample success from all sampled completions.")
    print(
        "Uniform audit bN labels come from run names; N is the audit budget per batch."
    )


if __name__ == "__main__":
    main()
