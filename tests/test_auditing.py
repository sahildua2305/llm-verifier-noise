import unittest

from pydantic import ValidationError

from src.auditing import Auditor, compute_audit_rate_metrics
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
        self.assertEqual(metrics["audit_candidate_prevalence"], -1.0)
        self.assertEqual(metrics["audit_rate"], -1.0)
        self.assertEqual(metrics["audit_candidate_coverage"], -1.0)

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


if __name__ == "__main__":
    unittest.main()
