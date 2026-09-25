import gzip
import hashlib
import io
import struct
import zipfile

import pytest
import requests

from munnet.remotezip import GZIP_HEADER, fetch_member_gz, gz_stream_sha256, list_members

CSV = ("region;oktmo;year;value\n" + "Алтайский край;01601000;2023;1\n" * 5000).encode()


class RangeResponse:
    def __init__(self, blob: bytes, start: int, end: int):
        self.data = blob[start : end + 1]
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class RangeServer:
    """Отдаёт куски файла по заголовку Range; fail_first имитирует разовые обрывы соединения."""

    def __init__(self, blob: bytes, fail_first: int = 0):
        self.blob, self.fail_first, self.requests = blob, fail_first, 0

    def get(self, url, headers, **kwargs):
        self.requests += 1
        if self.fail_first:
            self.fail_first -= 1
            raise requests.ConnectionError("reset")
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return RangeResponse(self.blob, start, min(end, len(self.blob) - 1))


def make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("other.csv", b"x" * 1000)
        zf.writestr("parts/data.csv", CSV)
    return buf.getvalue()


def stream_sha256(blob: bytes, info: zipfile.ZipInfo) -> str:
    name_len, extra_len = struct.unpack("<HH", blob[info.header_offset + 26 : info.header_offset + 30])
    start = info.header_offset + 30 + name_len + extra_len
    return hashlib.sha256(blob[start : start + info.compress_size]).hexdigest()


def test_member_is_saved_as_readable_gzip(tmp_path):
    blob = make_zip()
    server = RangeServer(blob)
    info = list_members(server, "https://x/a.zip", timeout=1)["parts/data.csv"]
    sha = stream_sha256(blob, info)
    dest = tmp_path / "data.csv.gz"
    fetch_member_gz(server, "https://x/a.zip", info, dest, sha, timeout=1)
    assert gzip.decompress(dest.read_bytes()) == CSV
    assert dest.read_bytes().startswith(GZIP_HEADER)
    assert gz_stream_sha256(dest) == sha


def test_wrong_sha256_leaves_no_file(tmp_path):
    server = RangeServer(make_zip())
    info = list_members(server, "https://x/a.zip", timeout=1)["parts/data.csv"]
    dest = tmp_path / "data.csv.gz"
    with pytest.raises(OSError, match="sha256"):
        fetch_member_gz(server, "https://x/a.zip", info, dest, "0" * 64, timeout=1)
    assert not dest.exists()


def test_range_requests_are_retried(tmp_path):
    blob = make_zip()
    info = list_members(RangeServer(blob), "https://x/a.zip", timeout=1)["parts/data.csv"]
    server = RangeServer(blob, fail_first=1)
    dest = tmp_path / "data.csv.gz"
    fetch_member_gz(server, "https://x/a.zip", info, dest, stream_sha256(blob, info), timeout=1)
    assert gzip.decompress(dest.read_bytes()) == CSV
