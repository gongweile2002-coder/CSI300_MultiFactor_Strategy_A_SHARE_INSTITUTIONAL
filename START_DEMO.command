#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  .venv/bin/python live.py demo
else
  python3 live.py demo
fi
if command -v open >/dev/null; then open outputs/v8_demo; fi
