import unittest

from pydantic import ValidationError

from src.auditing import Auditor, compute_audit_metrics, compute_audit_rate_metrics
from src.configs import AuditingConfig, AuditingStrategies


class AuditorTest(unittest.TestCase):
    def test_audit_rate_metrics_report_their_denominators(self):
        metrics = compute_audit_rate_metrics(
            candidates=8,
            audited=4,
            batch_size=64,
        )

        self.assertEqual(metrics["audit_batch_size"], 64)
        self.assertEqual(metrics["audit_candidate_prevalence"], 8 / 64)
        self.assertEqual(metrics["audit_rate"], 4 / 64)
        self.assertEqual(metrics["audit_candidate_coverage"], 4 / 8)

    def test_audit_rate_metrics_handle_empty_denominators(self):
        metrics = compute_audit_rate_metrics(
            candidates=0,
            audited=0,
            batch_size=0,
        )

        self.assertEqual(metrics["audit_batch_size"], 0)
        self.assertNotIn("audit_candidate_prevalence", metrics)
        self.assertNotIn("audit_rate", metrics)
        self.assertNotIn("audit_candidate_coverage", metrics)

    def test_no_audit_metrics_keep_counts_and_omit_undefined_ratios(self):
        metrics = compute_audit_metrics(
            candidates=3,
            audited=0,
            batch_size=4,
            audited_false_positives=0,
            false_positives_before=2,
            false_positives_after=2,
            audit_budget=0,
            cumulative_audited=0,
            cumulative_audited_false_positives=0,
            cumulative_excess_fp_found=0,
            exploration_count=0,
            exploitation_count=0,
        )

        self.assertEqual(metrics["audit_candidate_prevalence"], 0.75)
        self.assertEqual(metrics["audit_rate"], 0.0)
        self.assertEqual(metrics["false_positives_before_audit"], 2)
        self.assertEqual(metrics["false_positives_after_audit"], 2)
        self.assertEqual(metrics["audit_candidate_fp_rate"], 2 / 3)
        self.assertEqual(metrics["cumulative_excess_fp_found"], 0)
        self.assertEqual(metrics["post_audit_reward_contamination"], 2 / 3)
        self.assertNotIn("audit_precision", metrics)
        self.assertNotIn("audit_precision_lift", metrics)
        self.assertNotIn("audit_budget_utilization", metrics)
        self.assertNotIn("cumulative_audit_precision", metrics)

    def test_uniform_audits_only_verifier_positives_with_fixed_budget(self):
        auditor = Auditor(
            strategy=AuditingStrategies.UNIFORM,
            budget_per_batch=2,
            seed=7,
        )

        verifier_rewards = [1.0, 1.0, 0.0, 1.0]
        oracle_rewards = [0.0, 1.0, 1.0, 0.0]
        training_rewards = auditor.audit(verifier_rewards, oracle_rewards)

        self.assertEqual(auditor.audited_count, 2)
        self.assertEqual(sum(auditor.last_audit_mask), 2)
        self.assertFalse(auditor.last_audit_mask[2])
        for index, audited in enumerate(auditor.last_audit_mask):
            expected = oracle_rewards[index] if audited else verifier_rewards[index]
            self.assertEqual(training_rewards[index], expected)

    def test_uniform_is_reproducible_for_the_same_seed(self):
        first = Auditor(strategy="uniform", budget_per_batch=2, seed=11)
        second = Auditor(strategy="uniform", budget_per_batch=2, seed=11)

        first.audit([1.0] * 6, [0.0] * 6)
        second.audit([1.0] * 6, [0.0] * 6)

        self.assertEqual(first.last_audit_mask, second.last_audit_mask)

    def test_uniform_caps_budget_at_number_of_candidates(self):
        auditor = Auditor(strategy="uniform", budget_per_batch=5, seed=3)

        rewards = auditor.audit([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

        self.assertEqual(rewards, [0.0, 0.0, 0.0])
        self.assertEqual(auditor.candidate_count, 1)
        self.assertEqual(auditor.audited_count, 1)
        self.assertEqual(auditor.audited_false_positives, 1)
        self.assertEqual(auditor.false_positives_before, 1)
        self.assertEqual(auditor.false_positives_after, 0)
        self.assertEqual(auditor.exploration_count, 1)
        self.assertEqual(auditor.exploitation_count, 0)

    def test_audit_metrics_measure_budget_and_reward_correction(self):
        auditor = Auditor(strategy="uniform", budget_per_batch=3, seed=3)

        auditor.audit(
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        )

        self.assertEqual(auditor.batch_size, 4)
        self.assertEqual(auditor.audit_budget_utilization, 1.0)
        self.assertEqual(auditor.false_positive_recall, 1.0)
        self.assertAlmostEqual(auditor.cumulative_audit_precision, 2 / 3)
        self.assertEqual(auditor.post_audit_reward_contamination, 0.0)
        self.assertEqual(auditor.reward_correction_rate, 0.5)

    def test_audit_metrics_compare_precision_with_random_baseline(self):
        metrics = compute_audit_metrics(
            candidates=4,
            audited=2,
            batch_size=6,
            audited_false_positives=2,
            false_positives_before=2,
            false_positives_after=0,
            audit_budget=2,
            cumulative_audited=2,
            cumulative_audited_false_positives=2,
            cumulative_excess_fp_found=1,
            exploration_count=2,
            exploitation_count=0,
        )

        self.assertEqual(metrics["audit_candidate_fp_rate"], 0.5)
        self.assertEqual(metrics["audit_precision"], 1.0)
        self.assertEqual(metrics["audit_precision_lift"], 2.0)
        self.assertEqual(metrics["cumulative_excess_fp_found"], 1.0)

    def test_zero_candidate_fp_rate_omits_undefined_precision_lift(self):
        metrics = compute_audit_metrics(
            candidates=3,
            audited=2,
            batch_size=3,
            audited_false_positives=0,
            false_positives_before=0,
            false_positives_after=0,
            audit_budget=2,
            cumulative_audited=2,
            cumulative_audited_false_positives=0,
            cumulative_excess_fp_found=0,
            exploration_count=2,
            exploitation_count=0,
        )

        self.assertEqual(metrics["audit_candidate_fp_rate"], 0.0)
        self.assertEqual(metrics["audit_precision"], 0.0)
        self.assertNotIn("audit_precision_lift", metrics)

    def test_cumulative_excess_uses_each_batch_candidate_fp_rate(self):
        for strategy in ("uniform", "lexical"):
            with self.subTest(strategy=strategy):
                auditor = Auditor(
                    strategy=strategy,
                    budget_per_batch=2,
                    seed=3,
                    min_history=2,
                )
                texts = ["one", "two", "three", "four"]
                auditor.audit(
                    [1.0] * 4,
                    [0.0, 0.0, 1.0, 1.0],
                    texts if strategy == "lexical" else None,
                )
                expected = auditor.audited_false_positives - 2 * 0.5
                self.assertEqual(auditor.cumulative_excess_fp_found, expected)

                auditor.audit(
                    [1.0] * 4,
                    [0.0, 1.0, 1.0, 1.0],
                    texts if strategy == "lexical" else None,
                )
                expected += auditor.audited_false_positives - 2 * 0.25
                self.assertEqual(auditor.cumulative_excess_fp_found, expected)

    def test_disabled_auditing_preserves_rewards(self):
        auditor = Auditor()
        verifier_rewards = [1.0, 0.0, 1.0]

        self.assertEqual(
            auditor.audit(verifier_rewards, [0.0, 1.0, 1.0]), verifier_rewards
        )
        self.assertEqual(auditor.last_audit_mask, [False, False, False])

    def test_uniform_config_requires_positive_budget(self):
        with self.assertRaises(ValidationError):
            AuditingConfig(strategy="uniform", budget_per_batch=0)

    def test_lexical_config_validation_and_serialization(self):
        config = AuditingConfig(
            strategy="lexical",
            budget_per_batch=10,
            seed=42,
            k_neighbors=5,
            exploration_rate=0.2,
            min_history=10,
            max_history=1000,
        )

        self.assertEqual(config.strategy, AuditingStrategies.LEXICAL)
        self.assertEqual(config.to_dict()["min_history"], 10)
        with self.assertRaises(ValidationError):
            AuditingConfig(strategy="lexical", budget_per_batch=0)
        with self.assertRaises(ValidationError):
            AuditingConfig(
                strategy="lexical",
                budget_per_batch=1,
                min_history=10,
                max_history=9,
            )

    def test_lexical_b10_cold_start_then_activates(self):
        auditor = Auditor(
            strategy="lexical",
            budget_per_batch=10,
            seed=42,
            k_neighbors=5,
            exploration_rate=0.2,
            min_history=10,
        )
        texts = [f"shared arithmetic pattern {index}" for index in range(12)]

        auditor.audit([1.0] * 12, [index % 2 for index in range(12)], texts)

        self.assertEqual(auditor.audited_count, 10)
        self.assertEqual(auditor.exploration_count, 10)
        self.assertEqual(auditor.exploitation_count, 0)
        self.assertEqual(auditor.history_size, 10)

        auditor.audit([1.0] * 12, [1.0] * 12, texts)

        self.assertEqual(auditor.exploration_count, 2)
        self.assertEqual(auditor.exploitation_count, 8)
        self.assertEqual(auditor.lexical_candidate_risk_count, 12)
        self.assertEqual(sum(auditor.last_audit_mask), 10)

    def test_lexical_waits_until_history_exists_before_the_call(self):
        auditor = Auditor(
            strategy="lexical", budget_per_batch=10, min_history=10, seed=4
        )

        auditor.audit([1.0] * 6, [0.0] * 6, ["zorbax wrong"] * 6)
        self.assertEqual(auditor.exploration_count, 6)
        auditor.audit([1.0] * 5, [1.0] * 5, ["plain answer"] * 5)
        self.assertEqual(auditor.exploration_count, 5)
        self.assertEqual(auditor.history_size, 11)
        auditor.audit([1.0] * 5, [1.0] * 5, ["zorbax answer"] * 5)
        self.assertGreater(auditor.lexical_candidate_risk_count, 0)

    def test_lexical_similarity_learns_arbitrary_false_positive_markers(self):
        for marker in ("zorbax", "quuxle"):
            with self.subTest(marker=marker):
                auditor = Auditor(
                    strategy="lexical",
                    budget_per_batch=1,
                    seed=7,
                    k_neighbors=1,
                    exploration_rate=0.0,
                    min_history=2,
                )
                auditor.audit([1.0], [0.0], [f"{marker} calculation wrong"])
                auditor.audit([1.0], [1.0], ["ordinary correct response"])

                auditor.audit(
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [f"{marker} result wrong", "ordinary response"],
                )

                self.assertEqual(auditor.last_audit_mask, [True, False])
                self.assertGreater(
                    auditor.last_risk_scores[0], auditor.last_risk_scores[1]
                )
                self.assertEqual(auditor.last_selection_modes[0], "exploitation")

    def test_current_oracle_labels_do_not_change_lexical_selection(self):
        def primed() -> Auditor:
            auditor = Auditor(
                strategy="lexical",
                budget_per_batch=1,
                seed=11,
                k_neighbors=1,
                exploration_rate=0.0,
                min_history=2,
            )
            auditor.audit([1.0], [0.0], ["zorbax wrong answer"])
            auditor.audit([1.0], [1.0], ["careful correct answer"])
            return auditor

        first = primed()
        second = primed()
        texts = ["zorbax wrong result", "careful correct result"]
        first.audit([1.0, 1.0], [0.0, 1.0], texts)
        second.audit([1.0, 1.0], [1.0, 0.0], texts)

        self.assertEqual(first.last_audit_mask, second.last_audit_mask)

    def test_lexical_selection_is_deterministic_disjoint_and_budgeted(self):
        def run() -> Auditor:
            auditor = Auditor(
                strategy="lexical",
                budget_per_batch=10,
                seed=19,
                exploration_rate=0.2,
                min_history=10,
            )
            texts = [f"shared lexical text {index}" for index in range(20)]
            auditor.audit([1.0] * 20, [index % 2 for index in range(20)], texts)
            auditor.audit([1.0] * 20, [0.0] * 20, texts)
            return auditor

        first = run()
        second = run()

        self.assertEqual(first.last_audit_mask, second.last_audit_mask)
        self.assertEqual(sum(first.last_audit_mask), 10)
        selected_modes = [
            mode for mode in first.last_selection_modes if mode is not None
        ]
        self.assertEqual(selected_modes.count("exploration"), 2)
        self.assertEqual(selected_modes.count("exploitation"), 8)

    def test_lexical_history_contains_only_audits_and_keeps_most_recent(self):
        auditor = Auditor(
            strategy="lexical",
            budget_per_batch=2,
            min_history=1,
            max_history=3,
            seed=2,
        )
        auditor.audit([1.0, 1.0, 0.0], [0.0, 1.0, 0.0], ["a", "b", "c"])
        self.assertEqual(auditor.history_size, 2)
        self.assertNotIn("c", auditor.history_texts)
        auditor.audit([1.0, 1.0], [1.0, 1.0], ["d", "e"])

        self.assertEqual(auditor.history_size, 3)
        self.assertCountEqual(auditor.history_texts[-2:], ["d", "e"])

    def test_unusable_lexical_text_falls_back_to_uniform(self):
        auditor = Auditor(
            strategy="lexical",
            budget_per_batch=2,
            min_history=2,
            seed=5,
        )
        auditor.audit([1.0, 1.0], [0.0, 1.0], ["", ""])
        auditor.audit([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], ["", "", ""])

        self.assertEqual(auditor.exploration_count, 2)
        self.assertEqual(auditor.exploitation_count, 0)
        self.assertTrue(all(score is None for score in auditor.last_risk_scores))

    def test_lexical_requires_completion_text(self):
        auditor = Auditor(strategy="lexical", budget_per_batch=1)

        with self.assertRaisesRegex(ValueError, "completion_texts"):
            auditor.audit([1.0], [0.0])


if __name__ == "__main__":
    unittest.main()
