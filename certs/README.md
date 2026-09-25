# certs/ — временно, удалить после скачивания данных

Без этой папки Python и curl не открывают часть сайтов с данными конкурса:

| Сайт | Проблема | Что помогает |
|---|---|---|
| www.sberbank.ru, www.sberbank.com, rosstat.gov.ru | сертификат от УЦ Минцифры, его нет в стандартных хранилищах | `russian_trusted_ca.pem` |
| sberindex.ru | сервер не отдаёт промежуточный сертификат, а корня TrustAsia нет в хранилищах | `litessl_rsa_ca_2025.pem` + флаг `VERIFY_X509_PARTIAL_CHAIN` |
| sber.ru, sberindex.ru | отклоняют User-Agent `python-requests` | браузерный User-Agent |

Проверка TLS нигде не отключается. Отпечатки SHA-256 записаны в начале каждого `.pem`.

## Как пользоваться

Проверить все сайты (из корня репозитория):

```bash
python -m certs
```

Python:

```python
from certs import session

r = session().get("https://sberindex.ru/...", timeout=60)
r.raise_for_status()
```

curl:

```bash
cat "$SSL_CERT_FILE" certs/*.pem > /tmp/ca.pem
curl --cacert /tmp/ca.pem -A "Mozilla/5.0" https://sberindex.ru/
```

## Как удалить

1. Убедиться, что загрузчик данных и README проекта не импортируют `certs` (`grep -rn "certs" --include=*.py .`).
   Жюри будет запускать код без этой папки.
2. `git rm -r certs` и закоммитить.
3. Отметить пункт про удаление в `PLAN.md`.
