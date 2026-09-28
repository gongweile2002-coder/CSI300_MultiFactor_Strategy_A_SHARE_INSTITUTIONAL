
import argparse
import json
import os
from pathlib import Path

from src.tushare_provider import TushareDownloader

BASE = Path(__file__).resolve().parent

def load_dotenv_simple(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

def main():
    parser = argparse.ArgumentParser(description="Download real CSI300 research data from Tushare Pro.")
    parser.add_argument("--config", default=str(BASE/"config.example.json"))
    parser.add_argument("--max-stocks", type=int, default=None, help="Quick test mode, e.g. 30")
    parser.add_argument("--sleep", type=float, default=0.06)
    args = parser.parse_args()

    load_dotenv_simple(BASE/".env")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    token = os.getenv("TUSHARE_TOKEN", "")

    dl = TushareDownloader(
        token=token,
        output_dir=BASE/"data"/"real",
        sleep_seconds=args.sleep
    )
    summary = dl.download_all(
        index_code=cfg["index_code"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        max_stocks=args.max_stocks
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
