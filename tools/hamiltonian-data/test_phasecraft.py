from io import BytesIO
import unittest
from unittest.mock import patch

from fetch_phasecraft import RangeReader, SIZE


class PhasecraftRangeTests(unittest.TestCase):
    def test_cached_reads_preserve_seek_and_record_hash(self):
        response = BytesIO(b"a" * 65536)
        response.status = 206
        response.headers = {"Content-Range": f"bytes 0-65535/{SIZE}"}
        with patch("fetch_phasecraft.urllib.request.urlopen", return_value=response) as request:
            reader = RangeReader()
            self.assertEqual(reader.read(4), b"aaaa")
            reader.seek(17)
            self.assertEqual(reader.read(3), b"aaa")
            self.assertEqual(reader.tell(), 20)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(len(reader.ranges), 1)
            self.assertEqual(len(reader.ranges[0]["sha256"]), 64)

    def test_wrong_content_range_rejected(self):
        response = BytesIO(b"a" * 65536)
        response.status = 206
        response.headers = {"Content-Range": f"bytes 65536-131071/{SIZE}"}
        with patch("fetch_phasecraft.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "unexpected byte range"):
                RangeReader().read(2)

    def test_short_range_rejected(self):
        response = BytesIO(b"short")
        response.status = 206
        response.headers = {"Content-Range": f"bytes 0-65535/{SIZE}"}
        with patch("fetch_phasecraft.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "Wrong range length"):
                RangeReader().read(2)


if __name__ == "__main__":
    unittest.main()
