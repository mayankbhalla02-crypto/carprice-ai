#!/bin/sh
# Linux: run ./CarPriceAI.sh to start Car price AI.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 run.py
elif command -v python >/dev/null 2>&1; then
  python run.py
else
  echo
  echo "  Python 3 is not installed. Install it with your package manager"
  echo "  (for example: sudo apt install python3 python3-venv python3-pip)."
fi
