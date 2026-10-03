"""Сверка выходов двух прогонов: чистой копии репозитория и основной рабочей копии.

Так проверялся прогон с нуля 03.10.2026 (``docs/repro_2026-10-03.md``). Каждый файл сравнивается по sha256;
если байты различаются, файл разбирается по типу (json, csv, parquet, pkl, текст) и печатаются места, где
расходятся значения (числа — с допуском 1e-9). Рисунки (png, svg) сравниваются только побайтно.

Запуск: ``PYTHONIOENCODING=utf-8 uv run --frozen python scripts/compare_outputs.py <чистая копия>
<основная копия> <каталог или файл> [...] [--report отчёт.json]``; каталоги — относительно корней копий,
например ``outputs/cluster data/processed``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
MAX_ITEMS = 12


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short(v, n=90) -> str:
    try:
        s = json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def num_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
        b, (int, float, np.integer, np.floating)
    ):
        a, b = float(a), float(b)
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))
    return False


def deep_diff(a, b, path="", out=None):
    """Пути листьев, где значения расходятся (числа — с допуском TOL)."""
    if out is None:
        out = []
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        d = frame_diff(a, b)
        if d:
            out.append((path or "<frame>", d, ""))
        return out
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        d = frame_diff(a.to_frame(), b.to_frame())
        if d:
            out.append((path or "<series>", d, ""))
        return out
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            out.append((path, f"shape {a.shape}", f"shape {b.shape}"))
        elif a.dtype.kind in "fc" or b.dtype.kind in "fc":
            fa, fb = a.astype(float), b.astype(float)
            bad = ~(np.isclose(fa, fb, rtol=TOL, atol=TOL, equal_nan=True))
            if bad.any():
                out.append((path, f"{int(bad.sum())} элементов, max|Δ|={np.nanmax(np.abs(fa - fb)):.3g}", ""))
        elif not np.array_equal(a, b):
            out.append((path, "массивы различаются", ""))
        return out
    if dataclasses.is_dataclass(a) and dataclasses.is_dataclass(b) and not isinstance(a, type):
        da = {f.name: getattr(a, f.name) for f in dataclasses.fields(a)}
        db = {f.name: getattr(b, f.name) for f in dataclasses.fields(b)}
        return deep_diff(da, db, f"{path}<{type(a).__name__}>", out)
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            q = f"{path}.{k}"
            if k not in a:
                out.append((q, "<нет>", short(b[k])))
            elif k not in b:
                out.append((q, short(a[k]), "<нет>"))
            else:
                deep_diff(a[k], b[k], q, out)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, f"len {len(a)}", f"len {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            deep_diff(x, y, f"{path}[{i}]", out)
        return out
    if num_equal(a, b):
        return out
    try:
        same = a == b
        if isinstance(same, (bool, np.bool_)) and same:
            return out
    except Exception:
        pass
    out.append((path, short(a), short(b)))
    return out


def frame_diff(a: pd.DataFrame, b: pd.DataFrame) -> str:
    notes = []
    if a.shape != b.shape:
        notes.append(f"форма {a.shape} против {b.shape}")
    if list(a.columns) != list(b.columns):
        only_a = [c for c in a.columns if c not in b.columns]
        only_b = [c for c in b.columns if c not in a.columns]
        notes.append(f"колонки: только в чистой {only_a[:5]}, только в основной {only_b[:5]}")
    if a.shape[0] != b.shape[0]:
        return "; ".join(notes)
    if not a.index.equals(b.index):
        notes.append("индекс различается")
    for c in [c for c in a.columns if c in b.columns]:
        x, y = a[c], b[c]
        if (
            pd.api.types.is_numeric_dtype(x)
            and pd.api.types.is_numeric_dtype(y)
            and not pd.api.types.is_bool_dtype(x)
        ):
            fx, fy = x.to_numpy(dtype=float, na_value=np.nan), y.to_numpy(dtype=float, na_value=np.nan)
            bad = ~np.isclose(fx, fy, rtol=TOL, atol=TOL, equal_nan=True)
            if bad.any():
                i = int(np.argmax(bad))
                notes.append(
                    f"{c}: {int(bad.sum())} ячеек, max|Δ|={np.nanmax(np.abs(fx - fy)):.3g}, "
                    f"напр. строка {i}: {fx[i]!r} против {fy[i]!r}"
                )
        else:
            xs, ys = x.astype(str).to_numpy(), y.astype(str).to_numpy()
            bad = xs != ys
            if bad.any():
                i = int(np.argmax(bad))
                notes.append(
                    f"{c}: {int(bad.sum())} ячеек, напр. строка {i}: {xs[i][:60]!r} против {ys[i][:60]!r}"
                )
    return "; ".join(notes)


def read_table(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def compare_file(c: Path, m: Path) -> dict:
    ext = c.suffix.lower()
    res = {"kind": ext.lstrip(".")}
    if ext in (".png", ".svg", ".jpg", ".pdf", ".woff2", ".ttf"):
        res["status"] = "рисунок: байты различаются"
        if ext == ".svg":
            ca = c.read_text(encoding="utf-8", errors="replace").splitlines()
            ma = m.read_text(encoding="utf-8", errors="replace").splitlines()
            diff = [(i, x, y) for i, (x, y) in enumerate(zip(ca, ma, strict=False)) if x != y]
            res["details"] = [f"строк {len(ca)} против {len(ma)}; различных {len(diff)}"] + [
                f"стр. {i + 1}: {x.strip()[:70]!r} | {y.strip()[:70]!r}" for i, x, y in diff[:3]
            ]
        return res
    try:
        if ext == ".json":
            d = deep_diff(
                json.loads(c.read_text(encoding="utf-8")), json.loads(m.read_text(encoding="utf-8"))
            )
        elif ext in (".csv", ".parquet"):
            s = frame_diff(read_table(c), read_table(m))
            d = [("<таблица>", s, "")] if s else []
        elif ext == ".pkl":
            with c.open("rb") as f1, m.open("rb") as f2:
                d = deep_diff(pickle.load(f1), pickle.load(f2))
        else:
            ca = c.read_text(encoding="utf-8", errors="replace").splitlines()
            ma = m.read_text(encoding="utf-8", errors="replace").splitlines()
            diff = [(i, x, y) for i, (x, y) in enumerate(zip(ca, ma, strict=False)) if x != y]
            d = [(f"стр. {i + 1}", x.strip()[:100], y.strip()[:100]) for i, x, y in diff]
            if len(ca) != len(ma):
                d.append(("<число строк>", str(len(ca)), str(len(ma))))
    except Exception as e:  # noqa: BLE001
        res["status"] = f"не разобрать: {type(e).__name__}: {e}"
        return res
    if not d:
        res["status"] = f"совпадает по значениям (допуск {TOL:g}), байты различаются"
    else:
        res["status"] = f"РАЗЛИЧАЕТСЯ: {len(d)} мест"
        res["details"] = [f"{p}: {a} | {b}" if b else f"{p}: {a}" for p, a, b in d[:MAX_ITEMS]]
    return res


def main() -> None:
    args = sys.argv[1:]
    report_path = None
    if "--report" in args:
        i = args.index("--report")
        report_path = Path(args[i + 1])
        del args[i : i + 2]
    clean, main_root, targets = Path(args[0]), Path(args[1]), args[2:]
    report = {}
    for t in targets:
        cdir, mdir = clean / t, main_root / t
        if cdir.is_file() or mdir.is_file():
            pairs = [Path(t)]
        else:
            rel_c = {p.relative_to(clean) for p in cdir.rglob("*") if p.is_file()} if cdir.exists() else set()
            rel_m = (
                {p.relative_to(main_root) for p in mdir.rglob("*") if p.is_file()} if mdir.exists() else set()
            )
            pairs = sorted(rel_c | rel_m)
        counts = {
            "всего": 0,
            "побайтно": 0,
            "по значениям": 0,
            "рисунки разные": 0,
            "различаются": 0,
            "нет в чистой": 0,
            "нет в основной": 0,
            "не разобрать": 0,
        }
        files = {}
        for rel in pairs:
            counts["всего"] += 1
            c, m = clean / rel, main_root / rel
            if not c.exists():
                counts["нет в чистой"] += 1
                files[str(rel)] = {"status": "нет в чистой копии"}
                continue
            if not m.exists():
                counts["нет в основной"] += 1
                files[str(rel)] = {"status": "нет в основном репозитории"}
                continue
            if sha(c) == sha(m):
                counts["побайтно"] += 1
                continue
            r = compare_file(c, m)
            files[str(rel)] = r
            st = r["status"]
            if st.startswith("совпадает"):
                counts["по значениям"] += 1
            elif st.startswith("рисунок"):
                counts["рисунки разные"] += 1
            elif st.startswith("не разобрать"):
                counts["не разобрать"] += 1
            else:
                counts["различаются"] += 1
        report[t] = {"counts": counts, "files": files}
        print(f"\n## {t}: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v or k == "всего"))
        for rel, r in files.items():
            print(f"  - {rel}: {r['status']}")
            for line in r.get("details", []):
                print(f"      {line}")
    if report_path:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
