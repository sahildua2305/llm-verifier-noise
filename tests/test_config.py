import unittest

from pydantic import ValidationError

from src.configs import TrainConfig


def make_config(**overrides) -> TrainConfig:
    values = {
        "model_name": "test/model",
        "training_type": "grpo",
        "dataset": "rgym",
        "hf_username": "test-user",
        "training_args": {},
    }
    values.update(overrides)
    return TrainConfig(**values)


class TrainConfigRunNameTest(unittest.TestCase):
    def test_explicit_run_name_is_used_as_label(self):
        config = make_config(run_name="  clean-100-step  ")

        self.assertEqual(config.run_name, "clean-100-step")

    def test_run_name_is_optional_for_legacy_configs(self):
        config = make_config()

        self.assertIsNone(config.run_name)

    def test_run_name_cannot_create_nested_output_paths(self):
        with self.assertRaises(ValidationError):
            make_config(run_name="experiment/one")

    def test_lexical_auditing_rejects_checkpoint_resume(self):
        with self.assertRaisesRegex(ValidationError, "audit history"):
            make_config(
                auditing={"strategy": "lexical", "budget_per_batch": 10},
                resume_from_checkpoint="checkpoint-20",
            )


if __name__ == "__main__":
    unittest.main()
