# SPDX-License-Identifier: MIT
"""Deterministic writers: every output is written byte-for-byte reproducibly (sorted keys and rows,
shortest round-trip floats, fixed gzip/zip headers)."""
from __future__ import annotations
import csv, gzip, hashlib, io, json, math, zipfile
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import yaml

WRITER_ID = "lca.writer"
WRITER_SCHEMA_VERSION = "1"
GZIP_LEVEL = 9

def _decorate(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    out.setdefault("writer_id", WRITER_ID)
    out.setdefault("writer_schema_version", WRITER_SCHEMA_VERSION)
    return out

def canonical_json_bytes(obj: Any, *, decorate: bool = True) -> bytes:
    x = _decorate(obj) if decorate else obj
    return (json.dumps(x, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(obj, decorate=True))

def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        x = float(v)
        if not math.isfinite(x):
            raise ValueError("NaN/Inf forbidden")
        return repr(x)
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return str(v)

def _rows(rows: Iterable[dict[str, Any]], fields: list[str]) -> list[dict[str, str]]:
    fs = set(fields)
    out = []
    for row in rows:
        extra = set(row) - fs
        if extra:
            raise ValueError(f"undeclared CSV columns: {sorted(extra)}")
        out.append({k: _cell(row.get(k, "")) for k in fields})
    out.sort(key=lambda r: tuple(r[k] for k in fields))
    return out

def csv_bytes(rows: Iterable[dict[str, Any]], fields: list[str]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    w.writeheader(); w.writerows(_rows(rows, fields))
    return buf.getvalue().encode("utf-8")

def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes(rows, fields))

def write_gz_bytes(path: Path, raw: bytes, level: int = 9) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        with gzip.GzipFile(filename="", mode="wb", fileobj=fout, compresslevel=int(level), mtime=0) as gz:
            gz.write(raw)

def write_csv_gz(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> tuple[str, str]:
    raw = csv_bytes(rows, fields)
    write_gz_bytes(path, raw, 9)
    return hashlib.sha256(raw).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest()

def write_yaml(path: Path, obj: Any) -> None:
    txt = yaml.safe_dump(_decorate(obj), sort_keys=True, default_flow_style=False, allow_unicode=False)
    if not txt.endswith("\n"): txt += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8", newline="\n")

def _npy_bytes(a: np.ndarray) -> bytes:
    b = io.BytesIO(); np.lib.format.write_array(b, np.asarray(a), allow_pickle=False); return b.getvalue()

def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as zf:
        for name in sorted(arrays):
            zi = zipfile.ZipInfo(f"{name}.npy")
            zi.date_time = (1980,1,1,0,0,0); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o100644 & 0xFFFF) << 16; zi.create_system = 3; zi.flag_bits |= 0x800
            zf.writestr(zi, _npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"): text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path: Path, expected: str) -> Path:
    """Stop unless the file has the expected SHA-256 (inputs are read only after this check)."""
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"SHA-256 mismatch for {Path(path).name}: expected {expected}, got {got}")
    return Path(path)


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return sha256_bytes(str(a.dtype).encode() + b"|" + str(a.shape).encode() + b"|" + a.tobytes())
