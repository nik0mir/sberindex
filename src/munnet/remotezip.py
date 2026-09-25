"""Отдельные файлы из больших zip-архивов по HTTP Range, без скачивания архива целиком.

Сжатый поток внутри zip и внутри gzip один и тот же (deflate), поэтому член архива сохраняется как .gz
без перепаковки: байты остаются исходными, их sha256 можно закрепить в конфиге.
"""

import hashlib
import io
import struct
import time
import zipfile
import zlib
from pathlib import Path

import requests

# gzip: deflate, без имени файла и даты, чтобы результат не зависел от момента скачивания
GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"
GZIP_TRAILER_SIZE = 8


def get_range(session, url: str, start: int, end: int, timeout: float, tries: int = 5) -> bytes:
    """Байты [start, end] включительно. Хранилище иногда рвёт соединение, поэтому с повторами."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(tries):
        try:
            with session.get(url, headers=headers, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                data = b"".join(r.iter_content(1 << 20))
            if len(data) == end - start + 1:
                return data
            error: Exception = OSError(f"{url}: получено {len(data)} байт вместо {end - start + 1}")
        except requests.RequestException as e:
            error = e
        if attempt < tries - 1:
            time.sleep(2**attempt)
    raise error


class RangeFile(io.RawIOBase):
    """Удалённый файл только для чтения: каждое чтение — один запрос Range. Нужен zipfile для оглавления."""

    def __init__(self, session, url: str, timeout: float):
        self.session, self.url, self.timeout = session, url, timeout
        with session.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        buffer[:n] = get_range(self.session, self.url, self.pos, self.pos + n - 1, self.timeout)
        self.pos += n
        return n


def list_members(session, url: str, timeout: float) -> dict[str, zipfile.ZipInfo]:
    """Оглавление удалённого архива: читает только его конец."""
    with zipfile.ZipFile(RangeFile(session, url, timeout)) as zf:
        return {info.filename: info for info in zf.infolist()}


def _check_deflate(stream: bytes, crc: int, size: int, name: str) -> None:
    d = zlib.decompressobj(-zlib.MAX_WBITS)
    got_crc = got_size = 0
    for i in range(0, len(stream), 1 << 20):
        chunk = d.decompress(stream[i : i + (1 << 20)])
        got_crc, got_size = zlib.crc32(chunk, got_crc), got_size + len(chunk)
    chunk = d.flush()
    got_crc, got_size = zlib.crc32(chunk, got_crc), got_size + len(chunk)
    if (got_crc, got_size) != (crc, size):
        raise OSError(f"{name}: CRC или размер не совпали с оглавлением архива")


def fetch_member_gz(
    session, url: str, info: zipfile.ZipInfo, dest: Path, sha256: str, timeout: float
) -> None:
    """Сохраняет член архива в dest (.gz), сверив sha256 сжатого потока и CRC распакованных данных."""
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError(f"{info.filename}: поддерживается только сжатие deflate")
    header = get_range(session, url, info.header_offset, info.header_offset + 29, timeout)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    stream = get_range(session, url, start, start + info.compress_size - 1, timeout)
    got = hashlib.sha256(stream).hexdigest()
    if got != sha256:
        raise OSError(f"{info.filename}: sha256 {got}, ожидался {sha256}")
    _check_deflate(stream, info.CRC, info.file_size, info.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with open(part, "wb") as f:
        f.write(GZIP_HEADER)
        f.write(stream)
        f.write(struct.pack("<II", info.CRC, info.file_size & 0xFFFFFFFF))
    part.replace(dest)


def gz_stream_sha256(path: Path) -> str:
    """sha256 сжатого потока внутри .gz, записанного fetch_member_gz."""
    data = path.read_bytes()
    return hashlib.sha256(data[len(GZIP_HEADER) : -GZIP_TRAILER_SIZE]).hexdigest()
