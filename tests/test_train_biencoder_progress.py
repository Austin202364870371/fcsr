import unittest

from scripts.train_biencoder import create_training_progress


class _RecordingProgress:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfixes: list[dict[str, str]] = []
        self.closed = False

    def update(self, amount: int) -> None:
        self.updates.append(amount)

    def set_postfix(self, **kwargs: str) -> None:
        self.postfixes.append(kwargs)

    def close(self) -> None:
        self.closed = True


class BiEncoderProgressTests(unittest.TestCase):
    def test_creates_epoch_progress_with_loss_updates(self) -> None:
        progress = create_training_progress(
            epoch=2,
            total_batches=7,
            progress_factory=_RecordingProgress,
        )

        progress.update(1)
        progress.set_postfix(loss="0.1234")
        progress.close()

        self.assertEqual(progress.kwargs["total"], 7)
        self.assertEqual(progress.kwargs["desc"], "Bi-Encoder epoch 2")
        self.assertEqual(progress.kwargs["unit"], "batch")
        self.assertTrue(progress.kwargs["dynamic_ncols"])
        self.assertEqual(progress.updates, [1])
        self.assertEqual(progress.postfixes, [{"loss": "0.1234"}])
        self.assertTrue(progress.closed)


if __name__ == "__main__":
    unittest.main()
