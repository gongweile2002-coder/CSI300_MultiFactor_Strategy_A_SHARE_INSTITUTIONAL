
import argparse
from pathlib import Path
from src.reporting_v6 import generate_html_report

BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description="Generate standalone Quant Research HTML report.")
    p.add_argument("--folder", default=str(BASE/"outputs"/"demo_v6"))
    p.add_argument("--output", default=None)
    p.add_argument("--title", default="CSI300 Multi-Factor Quant Research Report")
    args = p.parse_args()

    folder = Path(args.folder)
    output = Path(args.output) if args.output else folder/"quant_research_report.html"
    path = generate_html_report(folder, output, args.title)
    print(f"Generated: {path}")

if __name__ == "__main__":
    main()
