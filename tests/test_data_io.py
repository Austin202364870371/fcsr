import gzip
import json
import tempfile
import unittest
from pathlib import Path

from data_io import JsonlError, load_jsonl, stream_jsonl, write_jsonl_atomic


class DataIoTests(unittest.TestCase):
    def test_reads_plain_gzip_and_sorted_directory_shards(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            (root / "b.jsonl").write_text('{"id": 2}\n', encoding="utf-8")
            with gzip.open(root / "a.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write('{"id": 1}\n')

            self.assertEqual(
                [item["id"] for item in stream_jsonl(root)],
                [1, 2],
            )
            self.assertEqual(load_jsonl(root / "b.jsonl"), [{"id": 2}])

    def test_reports_file_and_line_for_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(JsonlError, r"bad\.jsonl:2"):
                list(stream_jsonl(path))

    def test_atomic_writer_preserves_old_file_on_generator_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "output.jsonl"
            path.write_text('{"old": true}\n', encoding="utf-8")

            def records():
                yield {"new": 1}
                raise RuntimeError("stop")

            with self.assertRaisesRegex(RuntimeError, "stop"):
                write_jsonl_atomic(path, records())

            self.assertEqual(path.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())


if __name__ == "__main__":
    unittest.main()
