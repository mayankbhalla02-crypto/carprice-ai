#!/bin/bash
# macOS: double-click this file to start Car price AI.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  python3 run.py
elif command -v python >/dev/null 2>&1; then
  python run.py
else
  echo
  echo "  Python 3 is not installed. Opening the download page..."
  open "https://www.python.org/downloads/"
fi
echo
read -n 1 -s -r -p "  Press any key to close this window..."
