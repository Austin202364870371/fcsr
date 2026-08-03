import gzip
import tempfile
import unittest
from pathlib import Path

from scripts.migrate_jsonl_gzip import gzip_jsonl


class MigrateJsonlGzipTests(unittest.TestCase):
    def test_gzip_jsonl_preserves_decompressed_bytes_and_removes_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "records.jsonl"
            target = Path(directory) / "records.jsonl.gz"
            source.write_bytes(b'{"id":1}\n{"id":2}\n')

            result = gzip_jsonl(source, target, remove_source=True)

            self.assertFalse(source.exists())
            self.assertEqual(result.records, 2)
            self.assertEqual(result.source_sha256, result.decompressed_sha256)
            with gzip.open(target, "rb") as handle:
                self.assertEqual(handle.read(), b'{"id":1}\n{"id":2}\n')


if __name__ == "__main__":
    unittest.main()