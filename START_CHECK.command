#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
  .venv/bin/python live.py doctor
else
  python3 live.py doctor
fi
result=$?
if [ -f outputs/v9_doctor/doctor.md ]; then open outputs/v9_doctor/doctor.md; fi
if [ "$result" -ne 0 ]; then echo "请按 README 和检查报告完成本机配置；BLOCK 不代表已下单。"; fi
read -r -p "按回车退出：" response
