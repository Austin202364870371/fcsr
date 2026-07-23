import unittest

from scripts.train_reranker import (
    checkpoint_due,
    create_training_progress,
    enable_checkpoint_input_gradients,
    longest_group_index,
    resume_epoch_position,
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

    def test_checkpoint_waits_for_optimizer_boundary_after_interval(self) -> None:
        self.assertFalse(checkpoint_due(250, accumulation_count=10, next_due=250))
        self.assertTrue(checkpoint_due(256, accumulation_count=0, next_due=250))
        self.assertFalse(checkpoint_due(249, accumulation_count=0, next_due=250))

    def test_resume_epoch_position_uses_saved_order_only_for_saved_epoch(self) -> None:
        state = {"epoch": 0, "next_position": 32, "order": [3, 1, 0, 2]}
        self.assertEqual(resume_epoch_position(state, epoch=0), ([3, 1, 0, 2], 32))
        self.assertIsNone(resume_epoch_position(state, epoch=1))

    def test_longest_group_index_uses_largest_candidate_length(self) -> None:
        groups = [
            {"candidates": [{"prompt": "tiny"}]},
            {"candidates": [{"prompt": "medium"}, {"prompt": "longest"}]},
        ]
        processed: list[int] = []
        index, length = longest_group_index(
            groups,
            lambda text: len(text),
            progress=processed.append,
        )
        self.assertEqual((index, length), (1, 7))
        self.assertEqual(processed, [1, 1])


if __name__ == "__main__":
    unittest.main()