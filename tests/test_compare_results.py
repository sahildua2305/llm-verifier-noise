import unittest
from pathlib import Path

from eval_results.compare_results import Result, run_label, sort_key


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


if __name__ == "__main__":
    unittest.main()
