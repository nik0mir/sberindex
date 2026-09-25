"""Временный доступ к сайтам с нестандартными сертификатами: sberindex.ru, sberbank.ru/.com, rosstat.gov.ru.

Удалить вместе с папкой certs/, когда данные будут скачаны (см. certs/README.md).
"""
import ssl
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

CERTS_DIR = Path(__file__).resolve().parent
EXTRA_CA_FILES = sorted(CERTS_DIR.glob("*.pem"))

# sber.ru рвёт соединение, а sberindex.ru отвечает 403 на User-Agent python-requests.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


def ssl_context() -> ssl.SSLContext:
    """Системные корневые сертификаты плюс сертификаты из certs/."""
    ctx = ssl.create_default_context()
    for path in EXTRA_CA_FILES:
        ctx.load_verify_locations(cafile=path)
    # Корня LiteSSL RSA CA 2025 нет в хранилищах, поэтому доверяем промежуточному сертификату напрямую.
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx


class _Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = ssl_context()
        super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = ssl_context()
        return super().proxy_manager_for(*args, **kwargs)


def session() -> requests.Session:
    """requests.Session, которой открываются все сайты с данными конкурса."""
    s = requests.Session()
    s.headers["User-Agent"] = BROWSER_UA
    s.mount("https://", _Adapter())
    return s
