"""python -m certs — проверить доступ к сайтам с данными конкурса."""
import requests

from . import session

DOMAINS = [
    "sber.ru",
    "sberindex.ru",
    "www.sberbank.com",
    "www.sberbank.ru",
    "storage.yandexcloud.net",
    "tochno.st",
    "rosstat.gov.ru",
]

s = session()
for domain in DOMAINS:
    try:
        r = s.get(f"https://{domain}/", timeout=30)
        print(f"{domain:26} {r.status_code}")
    except requests.RequestException as e:
        print(f"{domain:26} ошибка: {type(e).__name__}: {str(e)[:120]}")
