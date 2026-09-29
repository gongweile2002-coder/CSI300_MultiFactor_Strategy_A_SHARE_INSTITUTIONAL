
import argparse
import json
import os
from pathlib import Path

from src.tushare_provider_v4 import TushareDownloaderV4

BASE = Path(__file__).resolve().parent

def load_dotenv_simple(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k,v=line.split("=",1)
        os.environ.setdefault(k.strip(),v.strip())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", default=str(BASE/"config.example.json"))
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.06)
    args=p.parse_args()

    load_dotenv_simple(BASE/".env")
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    dl=TushareDownloaderV4(
        token=os.getenv("TUSHARE_TOKEN",""),
        output_dir=BASE/"data"/"real",
        sleep_seconds=args.sleep
    )
    summary=dl.download_all_v4(
        cfg["index_code"],cfg["start_date"],cfg["end_date"],
        max_stocks=args.max_stocks
    )
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
