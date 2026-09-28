"""Readiness reports data truth; no synthetic PASS certificate."""
import sys
from live import main
if __name__=='__main__':raise SystemExit(main(['check']+sys.argv[1:]))
