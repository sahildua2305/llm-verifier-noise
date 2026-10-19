"""Strategy-neutral trusted-label auditing and reward correction."""

from dataclasses import dataclass, field

import numpy as np

from src.auditing.lexical_auditing import LexicalAuditor
from src.configs import AuditingStrategies


@dataclass
class Auditor:
    """Select verifier positives to audit and apply their trusted labels."""

    strategy: AuditingStrategies = AuditingStrategies.NONE
    budget_per_batch: int = 0
    seed: int = 42
    k_neighbors: int = 5
    exploration_rate: float = 0.2
    min_history: int = 10
    max_history: int = 1000

    candidate_count: int = field(default=0, init=False)
    batch_size: int = field(default=0, init=False)
    audited_count: int = field(default=0, init=False)
    audited_false_positives: int = field(default=0, init=False)
    false_positives_before: int = field(default=0, init=False)
    false_positives_after: int = field(default=0, init=False)
    exploration_count: int = field(default=0, init=False)
    exploitation_count: int = field(default=0, init=False)
    cumulative_audited_count: int = field(default=0, init=False)
    cumulative_audited_false_positives: int = field(default=0, init=False)
    last_audit_mask: list[bool] = field(default_factory=list, init=False)
    last_risk_scores: list[float | None] = field(default_factory=list, init=False)
    last_selection_modes: list[str | None] = field(default_factory=list, init=False)
    lexical_candidate_risk_sum: float = field(default=0.0, init=False)
    lexical_candidate_risk_count: int = field(default=0, init=False)
    lexical_selected_risk_sum: float = field(default=0.0, init=False)
    lexical_selected_risk_count: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _lexical: LexicalAuditor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str):
            self.strategy = AuditingStrategies(self.strategy)
        if self.budget_per_batch < 0:
            raise ValueError("budget_per_batch must be non-negative")
        if self.strategy != AuditingStrategies.NONE and self.budget_per_batch < 1:
            raise ValueError("active auditing requires budget_per_batch >= 1")
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be between 0 and 1")
        if self.min_history < 1:
            raise ValueError("min_history must be positive")
        if self.max_history < self.min_history:
            raise ValueError("max_history must be greater than or equal to min_history")

        self._rng = np.random.default_rng(self.seed)
        if self.strategy == AuditingStrategies.LEXICAL:
            self._lexical = LexicalAuditor(
                rng=self._rng,
                k_neighbors=self.k_neighbors,
                exploration_rate=self.exploration_rate,
                min_history=self.min_history,
                max_history=self.max_history,
            )

    @property
    def active(self) -> bool:
        return self.strategy != AuditingStrategies.NONE

    @property
    def history_size(self) -> int:
        return self._lexical.history_size if self._lexical is not None else 0

    @property
    def history_false_positives(self) -> int:
        if self._lexical is None:
            return 0
        return self._lexical.history_false_positives

    @property
    def history_texts(self) -> tuple[str, ...]:
        if self._lexical is None:
            return ()
        return tuple(self._lexical.history_texts)

    @property
    def audit_precision(self) -> float | None:
        if self.audited_count == 0:
            return None
        return self.audited_false_positives / self.audited_count

    @property
    def audit_budget_utilization(self) -> float | None:
        if self.budget_per_batch == 0:
            return None
        return self.audited_count / self.budget_per_batch

    @property
    def false_positive_recall(self) -> float | None:
        if self.false_positives_before == 0:
            return None
        return self.audited_false_positives / self.false_positives_before

    @property
    def cumulative_audit_precision(self) -> float | None:
        if self.cumulative_audited_count == 0:
            return None
        return self.cumulative_audited_false_positives / self.cumulative_audited_count

    @property
    def post_audit_reward_contamination(self) -> float | None:
        positive_rewards_after_audit = (
            self.candidate_count - self.audited_false_positives
        )
        if positive_rewards_after_audit == 0:
            return None
        return self.false_positives_after / positive_rewards_after_audit

    @property
    def reward_correction_rate(self) -> float | None:
        if self.batch_size == 0:
            return None
        return self.audited_false_positives / self.batch_size

    def _uniform_sample(self, indices: list[int], sample_size: int) -> list[int]:
        if sample_size == 0:
            return []
        return self._rng.choice(indices, size=sample_size, replace=False).tolist()

    def _reset_batch_stats(self, verifier: list[float]) -> None:
        self.batch_size = len(verifier)
        self.last_audit_mask = [False] * self.batch_size
        self.last_risk_scores = [None] * self.batch_size
        self.last_selection_modes = [None] * self.batch_size
        self.candidate_count = sum(value == 1.0 for value in verifier)
        self.audited_count = 0
        self.audited_false_positives = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.lexical_candidate_risk_sum = 0.0
        self.lexical_candidate_risk_count = 0
        self.lexical_selected_risk_sum = 0.0
        self.lexical_selected_risk_count = 0
        self.false_positives_before = 0

    def _select_lexical(
        self,
        candidate_indices: list[int],
        completion_texts: list[str],
        sample_size: int,
    ) -> list[int]:
        assert self._lexical is not None
        selection = self._lexical.select(
            candidate_indices=candidate_indices,
            completion_texts=completion_texts,
            sample_size=sample_size,
        )
        self.exploration_count = selection.exploration_count
        self.exploitation_count = selection.exploitation_count
        for index, risk in selection.risk_scores.items():
            self.last_risk_scores[index] = risk
        for index, mode in selection.selection_modes.items():
            self.last_selection_modes[index] = mode
        self.lexical_candidate_risk_sum = sum(selection.risk_scores.values())
        self.lexical_candidate_risk_count = len(selection.risk_scores)
        selected_risks = [
            selection.risk_scores[index]
            for index in selection.indices
            if index in selection.risk_scores
        ]
        self.lexical_selected_risk_sum = sum(selected_risks)
        self.lexical_selected_risk_count = len(selected_risks)
        return selection.indices

    def audit(
        self,
        verifier_rewards: list[float],
        oracle_rewards: list[bool | int | float],
        completion_texts: list[str] | None = None,
    ) -> list[float]:
        """Return rewards after auditing selected verifier-positive responses."""
        if len(verifier_rewards) != len(oracle_rewards):
            raise ValueError(
                "verifier_rewards and oracle_rewards must have equal length"
            )
        if completion_texts is not None and len(completion_texts) != len(
            verifier_rewards
        ):
            raise ValueError(
                "completion_texts and verifier_rewards must have equal length"
            )
        if self.strategy == AuditingStrategies.LEXICAL and completion_texts is None:
            raise ValueError("lexical auditing requires completion_texts")

        verifier = [float(value) for value in verifier_rewards]
        if any(value not in (0.0, 1.0) for value in verifier):
            raise ValueError("auditing requires binary verifier rewards")

        self._reset_batch_stats(verifier)
        candidate_indices = [
            index for index, reward in enumerate(verifier) if reward == 1.0
        ]
        sample_size = min(self.budget_per_batch, len(candidate_indices))
        selected: list[int] = []
        if self.strategy == AuditingStrategies.UNIFORM:
            selected = self._uniform_sample(candidate_indices, sample_size)
            self.exploration_count = len(selected)
            for index in selected:
                self.last_selection_modes[index] = "exploration"
        elif self.strategy == AuditingStrategies.LEXICAL:
            assert completion_texts is not None
            selected = self._select_lexical(
                candidate_indices, completion_texts, sample_size
            )
        elif self.strategy != AuditingStrategies.NONE:
            raise NotImplementedError(f"Unknown auditing strategy: {self.strategy}")

        # Current oracle labels are intentionally read only after selection.
        oracle = [float(value) for value in oracle_rewards]
        if any(value not in (0.0, 1.0) for value in oracle):
            raise ValueError("auditing requires binary oracle rewards")
        self.false_positives_before = sum(
            predicted == 1.0 and truth == 0.0
            for predicted, truth in zip(verifier, oracle)
        )

        training_rewards = verifier.copy()
        for index in selected:
            self.last_audit_mask[index] = True
            training_rewards[index] = oracle[index]

        self.audited_count = len(selected)
        self.audited_false_positives = sum(
            verifier[index] == 1.0 and oracle[index] == 0.0 for index in selected
        )
        self.false_positives_after = sum(
            predicted == 1.0 and truth == 0.0
            for predicted, truth in zip(training_rewards, oracle)
        )
        self.cumulative_audited_count += self.audited_count
        self.cumulative_audited_false_positives += self.audited_false_positives

        if self._lexical is not None:
            assert completion_texts is not None
            self._lexical.update_history(selected, completion_texts, oracle)
        return training_rewards
