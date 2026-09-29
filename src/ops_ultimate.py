
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
import pandas as pd


def sha256_file(path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    csv = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv).hexdigest()


def order_fingerprint(row) -> str:
    keys = [
        str(row.get("trade_date", "")),
        str(row.get("ticker", "")),
        str(row.get("side", "")),
        str(int(row.get("qty", 0))),
        f"{float(row.get('reference_price', 0.0)):.6f}",
    ]
    return hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()[:20]


def attach_order_fingerprints(orders: pd.DataFrame) -> pd.DataFrame:
    x = orders.copy()
    x["order_fingerprint"] = [order_fingerprint(r) for _, r in x.iterrows()]
    return x


def ensure_no_duplicate_orders(orders: pd.DataFrame):
    x = attach_order_fingerprints(orders)
    dup = x[x["order_fingerprint"].duplicated(keep=False)]
    if not dup.empty:
        raise ValueError(
            "Duplicate/idempotency collision detected for order fingerprints: "
            + ",".join(sorted(dup["order_fingerprint"].unique()))
        )
    return x


def kill_switch_path(base_dir, relative="ops/KILL_SWITCH") -> Path:
    return Path(base_dir) / relative


def kill_switch_is_on(base_dir, relative="ops/KILL_SWITCH") -> bool:
    return kill_switch_path(base_dir, relative).exists()


def set_kill_switch(base_dir, enabled: bool, relative="ops/KILL_SWITCH", reason="manual"):
    p = kill_switch_path(base_dir, relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    if enabled:
        p.write_text(
            json.dumps({
                "enabled": True,
                "reason": reason,
                "timestamp_utc": datetime.now(timezone.utc).isoformat()
            }, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    else:
        if p.exists():
            p.unlink()
    return p


def append_jsonl(path, event: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def git_commit_or_unknown(base_dir):
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=5
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return "unknown"


def build_run_manifest(base_dir, config_path=None, input_files=None, extra=None):
    base_dir = Path(base_dir)
    input_files = [Path(x) for x in (input_files or [])]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit_or_unknown(base_dir),
        "config_sha256": sha256_file(config_path) if config_path and Path(config_path).exists() else None,
        "inputs": {},
    }
    for p in input_files:
        if p.exists():
            manifest["inputs"][str(p)] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }
    if extra:
        manifest["extra"] = extra
    return manifest


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def approve_orders(orders: pd.DataFrame, approved_by: str, approval_note="manual review"):
    x = orders.copy()
    x["approved"] = True
    x["approved_by"] = approved_by
    x["approval_note"] = approval_note
    x["approval_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return x
