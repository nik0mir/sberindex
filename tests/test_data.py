import hashlib
import zipfile

import pytest
import requests

from munnet.data import _zip_member_name, extract, fetch, sha256


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class FakeSession:
    """Отдаёт по URL заранее заданный ответ; исключение в ответах имитирует сетевую ошибку."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


def source_for(content: bytes, *urls: str) -> dict:
    return {"sha256": hashlib.sha256(content).hexdigest(), "urls": list(urls)}


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    assert sha256(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_fetch_skips_network_when_file_is_valid(tmp_path):
    dest = tmp_path / "a.zip"
    dest.write_bytes(b"data")
    session = FakeSession({})
    assert fetch(source_for(b"data", "https://x"), dest, session, timeout=1) == dest
    assert session.calls == []


def test_fetch_falls_back_to_mirror(tmp_path):
    dest = tmp_path / "a.zip"
    session = FakeSession({"https://main": requests.ConnectionError("down"), "https://mirror": b"data"})
    fetch(source_for(b"data", "https://main", "https://mirror"), dest, session, timeout=1)
    assert dest.read_bytes() == b"data"
    assert session.calls == ["https://main", "https://mirror"]


def test_fetch_rejects_wrong_hash(tmp_path):
    dest = tmp_path / "a.zip"
    session = FakeSession({"https://main": b"tampered"})
    with pytest.raises(RuntimeError, match="sha256"):
        fetch(source_for(b"data", "https://main"), dest, session, timeout=1)
    assert not dest.exists()
    assert not (tmp_path / "a.zip.part").exists()


@pytest.mark.parametrize(
    "stored",
    [
        "Данные.pdf".encode("cp866").decode("cp437"),  # архив из Windows, Python 3.11
        "Данные.pdf".encode().decode("cp437"),  # UTF-8 без флага
        "Данные.pdf",  # Python 3.12+ прочитал поле Unicode Path
    ],
)
def test_zip_member_name_recovers_cyrillic(stored):
    info = zipfile.ZipInfo("x")
    info.filename, info.flag_bits = stored, 0
    assert _zip_member_name(info) == "Данные.pdf"


def test_extract_zip_flattens_folders_and_keeps_cyrillic_names(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/data.parquet", b"1")
        zf.writestr("inner/Описание.pdf", b"2")
    extract(archive, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["data.parquet", "Описание.pdf"]
