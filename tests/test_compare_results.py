import unittest
from pathlib import Path

from eval_results.compare_results import (
    Result,
    replicate_summary_rows,
    run_label,
    run_seed,
    sort_key,
)


def make_result(run: str) -> Result:
    return Result(
        path=Path("benchmark") / run / "result.json",
        benchmark="benchmark",
        run=run,
        category="category",
        dataset="dataset",
        count=1,
        metrics={"pass_at_1": 0.0, "pass_at_5": 0.0, "pass_at_16": 0.0},
        settings={},
        questions=(),
    )


class CompareResultsTest(unittest.TestCase):
    def test_lexical_run_names_are_labeled_and_ordered_after_uniform(self):
        root = Path("/tmp/results")
        lexical = make_result(
            "runs-20261018_dsum-certainly-lexical-audit-b10-checkpoint-60"
        )
        uniform = make_result(
            "runs-20261018_dsum-certainly-uniform-audit-b10-checkpoint-60"
        )
        lexical.path = root / lexical.path
        uniform.path = root / uniform.path
        baseline_paths = [Path("base.json"), Path("clean.json"), Path("flawed.json")]

        self.assertEqual(run_label(lexical, root, *baseline_paths), "Lexical audit b10")
        self.assertLess(
            sort_key(uniform, root, *baseline_paths),
            sort_key(lexical, root, *baseline_paths),
        )

    def test_seed_suffix_is_parsed_and_legacy_runs_default_to_42(self):
        root = Path("/tmp/results")
        legacy = make_result(
            "runs-20261018_dsum-certainly-lexical-audit-b10-checkpoint-60"
        )
        replicated = make_result(
            "runs-20261019_dsum-certainly-lexical-audit-b10-s43-checkpoint-60"
        )
        legacy.path = root / legacy.path
        replicated.path = root / replicated.path
        baseline_paths = [Path("base.json"), Path("clean.json"), Path("flawed.json")]

        self.assertEqual(run_seed(legacy, root, *baseline_paths), 42)
        self.assertEqual(run_seed(replicated, root, *baseline_paths), 43)
        self.assertLess(
            sort_key(legacy, root, *baseline_paths),
            sort_key(replicated, root, *baseline_paths),
        )

    def test_replicate_summary_reports_mean_and_sample_sd_across_seeds(self):
        root = Path("/tmp/results")
        seed_42 = make_result(
            "runs-20261018_dsum-certainly-lexical-audit-b10-checkpoint-60"
        )
        seed_43 = make_result(
            "runs-20261019_dsum-certainly-lexical-audit-b10-s43-checkpoint-60"
        )
        seed_42.path = root / seed_42.path
        seed_43.path = root / seed_43.path
        seed_42.metrics = {
            "pass_at_1": 0.6,
            "pass_at_5": 0.8,
            "pass_at_16": 0.9,
        }
        seed_43.metrics = {
            "pass_at_1": 0.8,
            "pass_at_5": 0.9,
            "pass_at_16": 1.0,
        }
        baseline_paths = [Path("base.json"), Path("clean.json"), Path("flawed.json")]

        summaries = replicate_summary_rows(
            [seed_42, seed_43], root, *baseline_paths
        )

        self.assertEqual(
            summaries,
            [
                [
                    "Lexical audit b10",
                    "42, 43",
                    "2",
                    "70.00% ± 14.14%",
                    "85.00% ± 7.07%",
                    "95.00% ± 7.07%",
                ]
            ],
        )


if __name__ == "__main__":
    unittest.main()
