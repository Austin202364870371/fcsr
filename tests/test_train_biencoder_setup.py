import unittest

from scripts.train_biencoder import enable_checkpoint_input_gradients


class _ModelWithInputGradientSupport:
    def __init__(self) -> None:
        self.calls = 0

    def enable_input_require_grads(self) -> None:
        self.calls += 1


class BiEncoderSetupTests(unittest.TestCase):
    def test_enables_input_gradients_before_checkpointed_lora_training(self) -> None:
        model = _ModelWithInputGradientSupport()

        enable_checkpoint_input_gradients(model)

        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
