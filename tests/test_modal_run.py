import unittest

from modal_run import _evaluation_result_path, _normalize_model_name


class ModalEvaluationPathTest(unittest.TestCase):
    def test_trailing_slash_is_removed_from_model_and_result_paths(self):
        model_name = "/runs/example/checkpoint-60/"

        self.assertEqual(
            _normalize_model_name(model_name), "/runs/example/checkpoint-60"
        )
        self.assertEqual(
            _evaluation_result_path(model_name),
            "RGYM_decimal_chain_sum/runs-example-checkpoint-60",
        )

    def test_empty_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            _evaluation_result_path("///")


if __name__ == "__main__":
    unittest.main()
