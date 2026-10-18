"""Trusted-label auditing for verifier rewards."""

from dataclasses import dataclass, field

import numpy as np

from src.configs import AuditingStrategies


@dataclass
class Auditor:
    """Audit verifier-positive responses and replace their rewards with oracle labels."""

    strategy: AuditingStrategies = AuditingStrategies.NONE
    budget_per_batch: int = 0
    seed: int = 42

    candidate_count: int = field(default=0, init=False)
    batch_size: int = field(default=0, init=False)
    audited_count: int = field(default=0, init=False)
    audited_false_positives: int = field(default=0, init=False)
    false_positives_before: int = field(default=0, init=False)
    false_positives_after: int = field(default=0, init=False)
    cumulative_audited_count: int = field(default=0, init=False)
    cumulative_audited_false_positives: int = field(default=0, init=False)
    last_audit_mask: list[bool] = field(default_factory=list, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            self.strategy = AuditingStrategies(self.strategy)
        if self.budget_per_batch < 0:
            raise ValueError("budget_per_batch must be non-negative")
        if self.strategy == AuditingStrategies.UNIFORM and self.budget_per_batch < 1:
            raise ValueError("uniform auditing requires budget_per_batch >= 1")
        self._rng = np.random.default_rng(self.seed)

    @property
    def active(self) -> bool:
        return self.strategy != AuditingStrategies.NONE

    @property
    def audit_precision(self) -> float:
        if self.audited_count == 0:
            return -1.0
        return self.audited_false_positives / self.audited_count

    @property
    def audit_budget_utilization(self) -> float:
        if self.budget_per_batch == 0:
            return -1.0
        return self.audited_count / self.budget_per_batch

    @property
    def false_positive_recall(self) -> float:
        if self.false_positives_before == 0:
            return -1.0
        return self.audited_false_positives / self.false_positives_before

    @property
    def cumulative_audit_precision(self) -> float:
        if self.cumulative_audited_count == 0:
            return -1.0
        return self.cumulative_audited_false_positives / self.cumulative_audited_count

    @property
    def post_audit_reward_contamination(self) -> float:
        positive_rewards_after_audit = (
            self.candidate_count - self.audited_false_positives
        )
        if positive_rewards_after_audit == 0:
            return -1.0
        return self.false_positives_after / positive_rewards_after_audit

    @property
    def reward_correction_rate(self) -> float:
        if self.batch_size == 0:
            return -1.0
        return self.audited_false_positives / self.batch_size

    def audit(
        self,
        verifier_rewards: list[float],
        oracle_rewards: list[bool | int | float],
    ) -> list[float]:
        """Return training rewards after auditing selected verifier positives.

        Uniform auditing samples without replacement from responses accepted by
        the verifier. Each selected reward is replaced by its trusted oracle label.
        """
        if len(verifier_rewards) != len(oracle_rewards):
            raise ValueError(
                "verifier_rewards and oracle_rewards must have equal length"
            )

        verifier = [float(value) for value in verifier_rewards]
        oracle = [float(value) for value in oracle_rewards]
        if any(value not in (0.0, 1.0) for value in verifier):
            raise ValueError("uniform auditing requires binary verifier rewards")
        if any(value not in (0.0, 1.0) for value in oracle):
            raise ValueError("uniform auditing requires binary oracle rewards")

        self.batch_size = len(verifier)
        self.last_audit_mask = [False] * self.batch_size
        self.candidate_count = sum(value == 1.0 for value in verifier)
        self.audited_count = 0
        self.audited_false_positives = 0
        self.false_positives_before = sum(
            predicted == 1.0 and truth == 0.0
            for predicted, truth in zip(verifier, oracle)
        )

        if not self.active:
            self.false_positives_after = self.false_positives_before
            return verifier

        if self.strategy != AuditingStrategies.UNIFORM:
            raise NotImplementedError(f"Unknown auditing strategy: {self.strategy}")

        candidate_indices = [
            index for index, reward in enumerate(verifier) if reward == 1.0
        ]
        sample_size = min(self.budget_per_batch, len(candidate_indices))
        if sample_size:
            selected = self._rng.choice(
                candidate_indices, size=sample_size, replace=False
            ).tolist()
        else:
            selected = []

        training_rewards = verifier.copy()
        for index in selected:
            self.last_audit_mask[index] = True
            training_rewards[index] = oracle[index]

        self.audited_count = sample_size
        self.audited_false_positives = sum(
            verifier[index] == 1.0 and oracle[index] == 0.0 for index in selected
        )
        self.false_positives_after = sum(
            predicted == 1.0 and truth == 0.0
            for predicted, truth in zip(training_rewards, oracle)
        )
        self.cumulative_audited_count += self.audited_count
        self.cumulative_audited_false_positives += self.audited_false_positives
        return training_rewards
