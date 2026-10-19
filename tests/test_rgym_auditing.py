import re
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from src.auditing import Auditor


def load_reasoning_gym_class():
    """Load the reward path without importing the broken local pyarrow build."""
    data_package = ModuleType("src.data")
    data_package.__path__ = [str(Path(__file__).parents[1] / "src" / "data")]
    base_module = ModuleType("src.data.base")
    base_module.MixedUpDataset = object
    base_module.Mixup = object
    datasets_module = ModuleType("datasets")
    datasets_module.concatenate_datasets = lambda *args, **kwargs: None
    datasets_module.load_dataset = lambda *args, **kwargs: None
    reasoning_gym_module = ModuleType("reasoning_gym")
    reasoning_gym_module.__path__ = []
    reasoning_gym_module.create_dataset = lambda *args, **kwargs: None
    reasoning_gym_composite = ModuleType("reasoning_gym.composite")
    reasoning_gym_composite.DatasetSpec = object
    reasoning_gym_dataset = ModuleType("reasoning_gym.dataset")
    reasoning_gym_dataset.ProceduralDataset = object
    math_verify_module = ModuleType("math_verify")
    math_verify_module.parse = lambda value: value
    math_verify_module.verify = lambda expected, actual: expected == actual
    transformers_module = ModuleType("transformers")
    transformers_module.PreTrainedTokenizerBase = object
    utils_module = ModuleType("src.utils")
    utils_module.extract_boxed = lambda text: re.search(
        r"\\boxed\{([^}]*)\}", text
    ).group(1)
    utils_module.process_ans = lambda value: value
    with patch.dict(
        "sys.modules",
        {
            "src.data": data_package,
            "src.data.base": base_module,
            "datasets": datasets_module,
            "reasoning_gym": reasoning_gym_module,
            "reasoning_gym.composite": reasoning_gym_composite,
            "reasoning_gym.dataset": reasoning_gym_dataset,
            "math_verify": math_verify_module,
            "transformers": transformers_module,
            "src.utils": utils_module,
        },
    ):
        from src.data.rgym import ReasoningGym

    return ReasoningGym


class _ScoringDataset:
    def score_answer(self, answer, entry):
        return float(answer == entry["answer"])


class _RawDataset:
    dataset = _ScoringDataset()


class _Mixup:
    def __init__(self, auditor):
        self.auditor = auditor

    def mixup_rewards(self, rewards, items):
        return [1.0] * len(rewards)


class ReasoningGymAuditingIntegrationTest(unittest.TestCase):
    def test_reward_path_passes_completion_text_and_exports_debug_buffers(self):
        ReasoningGym = load_reasoning_gym_class()
        dataset = ReasoningGym.__new__(ReasoningGym)
        dataset.raw_dataset = _RawDataset()
        dataset.mixup = _Mixup(
            Auditor(
                strategy="lexical",
                budget_per_batch=2,
                min_history=2,
                seed=42,
            )
        )
        _, reward = dataset.prepare_for_grpo()
        completions = [
            [{"content": "Zorbax, the result is \\boxed{3}."}],
            [{"content": "The result is \\boxed{4}."}],
        ]
        items = [
            {"question": "q1", "answer": "2", "metadata": {}},
            {"question": "q2", "answer": "4", "metadata": {}},
        ]

        rewards = reward(completions, item=items)

        self.assertEqual(rewards, [0.0, 1.0])
        self.assertEqual(
            dataset.mixup.auditor.history_texts,
            tuple(completion[0]["content"] for completion in completions),
        )
        self.assertEqual(reward._audit_mask, [1.0, 1.0])
        self.assertEqual(len(reward._audit_risk_scores), 2)
        self.assertEqual(reward._audit_selection_modes, ["exploration", "exploration"])


if __name__ == "__main__":
    unittest.main()
