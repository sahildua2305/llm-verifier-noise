"""TF-IDF nearest-neighbor selection for lexical adaptive auditing."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class LexicalSelection:
    """Selection result produced without current-batch oracle labels."""

    indices: list[int]
    risk_scores: dict[int, float]
    selection_modes: dict[int, str]
    exploration_count: int
    exploitation_count: int


@dataclass
class LexicalAuditor:
    """Rank verifier-positive completions using previously audited examples."""

    rng: np.random.Generator
    k_neighbors: int = 5
    exploration_rate: float = 0.2
    min_history: int = 10
    max_history: int = 1000
    history_texts: list[str] = field(default_factory=list, init=False)
    history_false_positive_labels: list[int] = field(default_factory=list, init=False)

    @property
    def history_size(self) -> int:
        return len(self.history_texts)

    @property
    def history_false_positives(self) -> int:
        return sum(self.history_false_positive_labels)

    def _uniform_sample(self, indices: list[int], sample_size: int) -> list[int]:
        if sample_size == 0:
            return []
        return self.rng.choice(indices, size=sample_size, replace=False).tolist()

    def _estimate_risks(
        self, candidate_indices: list[int], completion_texts: list[str]
    ) -> dict[int, float] | None:
        if self.history_size < self.min_history:
            return None

        candidate_texts = [completion_texts[index] for index in candidate_indices]
        try:
            matrix = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
            ).fit_transform([*self.history_texts, *candidate_texts])
        except ValueError:
            return None

        history_matrix = matrix[: self.history_size]
        candidate_matrix = matrix[self.history_size :]
        similarities = cosine_similarity(candidate_matrix, history_matrix)
        if similarities.size == 0 or not np.any(similarities > 0):
            return None

        labels = np.asarray(self.history_false_positive_labels, dtype=float)
        neighbor_count = min(self.k_neighbors, self.history_size)
        risks: dict[int, float] = {}
        for index, row in zip(candidate_indices, similarities):
            nearest = np.argsort(-row, kind="stable")[:neighbor_count]
            risks[index] = float(labels[nearest].mean())
        return risks

    def select(
        self,
        candidate_indices: list[int],
        completion_texts: list[str],
        sample_size: int,
    ) -> LexicalSelection:
        """Select audits without consulting current-batch oracle labels."""
        risks = self._estimate_risks(candidate_indices, completion_texts)
        if risks is None:
            selected = self._uniform_sample(candidate_indices, sample_size)
            return LexicalSelection(
                indices=selected,
                risk_scores={},
                selection_modes={index: "exploration" for index in selected},
                exploration_count=len(selected),
                exploitation_count=0,
            )

        exploration_count = min(
            sample_size, int(round(sample_size * self.exploration_rate))
        )
        exploration = self._uniform_sample(candidate_indices, exploration_count)
        exploration_set = set(exploration)
        remaining = [
            index for index in candidate_indices if index not in exploration_set
        ]
        tie_breakers = self.rng.random(len(remaining))
        ranked = sorted(
            zip(remaining, tie_breakers),
            key=lambda pair: (-risks[pair[0]], pair[1]),
        )
        exploitation = [index for index, _ in ranked[: sample_size - exploration_count]]
        modes = {index: "exploration" for index in exploration}
        modes.update({index: "exploitation" for index in exploitation})
        return LexicalSelection(
            indices=[*exploration, *exploitation],
            risk_scores=risks,
            selection_modes=modes,
            exploration_count=len(exploration),
            exploitation_count=len(exploitation),
        )

    def update_history(
        self,
        selected: list[int],
        completion_texts: list[str],
        oracle_rewards: list[float],
    ) -> None:
        """Add trusted labels only after the current selection is complete."""
        self.history_texts.extend(completion_texts[index] for index in selected)
        self.history_false_positive_labels.extend(
            int(oracle_rewards[index] == 0.0) for index in selected
        )
        overflow = self.history_size - self.max_history
        if overflow > 0:
            del self.history_texts[:overflow]
            del self.history_false_positive_labels[:overflow]
