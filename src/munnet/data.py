"""Этап data: скачивание исходных данных с проверкой sha256 в data/raw/<источник>/.

Архивы zip и rar распаковываются рядом с собой. Для источников kind: zip_members из больших архивов
берутся только перечисленные файлы (см. munnet.remotezip).
"""

import hashlib
import logging
import ssl
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from munnet.config import Config
from munnet.remotezip import fetch_member_gz, gz_stream_sha256, list_members

log = logging.getLogger(__name__)

EXTRACTED_MARK = ".extracted_sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _TLSAdapter(HTTPAdapter):
    """Системные корневые сертификаты плюс ca_files из конфига."""

    def __init__(self, ca_files: list[str]):
        self._ctx = ssl.create_default_context()
        for path in ca_files:
            self._ctx.load_verify_locations(cafile=path)
        # sberindex.ru не отдаёт промежуточный сертификат, а его корня нет в хранилищах:
        # доверяем промежуточному напрямую.
        self._ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        super().__init__()

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ctx
        return super().proxy_manager_for(*args, **kwargs)


def make_session(download_cfg: dict) -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = download_cfg["user_agent"]
    session.mount("https://", _TLSAdapter(download_cfg.get("ca_files", [])))
    return session


def fetch(source: dict, dest: Path, session: requests.Session, timeout: float) -> Path:
    """Скачивает файл, перебирая зеркала. Если файл уже есть и хеш совпал, сеть не трогает."""
    if dest.exists() and sha256(dest) == source["sha256"]:
        log.info("%s уже скачан", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in source["urls"]:
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except requests.RequestException as e:
            errors.append(f"{url}: {e}")
            continue
        got = sha256(part)
        if got != source["sha256"]:
            part.unlink()
            errors.append(f"{url}: sha256 {got}, ожидался {source['sha256']}")
            continue
        part.replace(dest)
        log.info("%s скачан с %s", dest.name, url)
        return dest
    raise RuntimeError(f"Не удалось скачать {dest.name}:\n" + "\n".join(errors))


def _zip_member_name(info: zipfile.ZipInfo) -> str:
    """Имя файла в архиве: без флага UTF-8 zipfile читает его как cp437, а архивы бывают в UTF-8 и cp866."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:
        # Python 3.12+ сам берёт имя из поля Unicode Path, если оно есть в архиве.
        return info.filename
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp866")


def extract(archive: Path, dest_dir: Path) -> None:
    """Распаковывает zip или rar в dest_dir без вложенных папок."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    (dest_dir / Path(_zip_member_name(info)).name).write_bytes(zf.read(info))
    elif archive.suffix == ".rar":
        import libarchive  # нужна системная libarchive, см. README

        with libarchive.file_reader(str(archive)) as entries:
            for entry in entries:
                if entry.isfile:
                    with open(dest_dir / Path(entry.pathname).name, "wb") as f:
                        for block in entry.get_blocks():
                            f.write(block)
    else:
        raise ValueError(f"Неизвестный формат архива: {archive.name}")


def extract_once(archive: Path, dest_dir: Path, archive_sha256: str) -> None:
    mark = dest_dir / EXTRACTED_MARK
    if mark.exists() and mark.read_text() == archive_sha256:
        log.info("%s уже распакован в %s", archive.name, dest_dir)
        return
    extract(archive, dest_dir)
    mark.write_text(archive_sha256)
    log.info("%s распакован в %s", archive.name, dest_dir)


def fetch_members(source: dict, dest_dir: Path, session: requests.Session, timeout: float) -> None:
    """Нужные файлы из больших zip-архивов: каждый сохраняется как dest_dir/<имя>.csv.gz."""
    directories: dict[str, dict] = {}
    for name, member in source["members"].items():
        dest = dest_dir / f"{name}.csv.gz"
        if dest.exists() and gz_stream_sha256(dest) == member["sha256"]:
            log.info("%s уже скачан", dest.name)
            continue
        url = source["base_url"] + member["archive"]
        if url not in directories:
            directories[url] = list_members(session, url, timeout)
        fetch_member_gz(session, url, directories[url][member["member"]], dest, member["sha256"], timeout)
        log.info("%s скачан из %s", dest.name, member["archive"])


def run(cfg: Config) -> None:
    raw = cfg.dir("raw")
    session = make_session(cfg["download"])
    timeout = cfg["download"]["timeout"]
    for name, source in cfg["sources"].items():
        dest_dir = raw / name
        if source.get("kind") == "zip_members":
            fetch_members(source, dest_dir, session, timeout)
            continue
        path = fetch(source, dest_dir / source["file"], session, timeout)
        if path.suffix in (".zip", ".rar"):
            extract_once(path, dest_dir, source["sha256"])
