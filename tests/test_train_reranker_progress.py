import unittest

from scripts.train_reranker import (
    create_training_progress,
    enable_checkpoint_input_gradients,
)


class _RecordingProgress:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _ModelWithInputGradientSupport:
    def __init__(self) -> None:
        self.calls = 0

    def enable_input_require_grads(self) -> None:
        self.calls += 1


class RerankerSetupTests(unittest.TestCase):
    def test_creates_group_progress_and_enables_checkpoint_input_gradients(self) -> None:
        progress = create_training_progress(
            epoch=1,
            total_groups=9,
            progress_factory=_RecordingProgress,
        )
        model = _ModelWithInputGradientSupport()
        enable_checkpoint_input_gradients(model)

        self.assertEqual(progress.kwargs["total"], 9)
        self.assertEqual(progress.kwargs["desc"], "Reranker epoch 1")
        self.assertEqual(progress.kwargs["unit"], "group")
        self.assertTrue(progress.kwargs["dynamic_ncols"])
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()