#!/usr/bin/env python3
"""One-step launcher for Windows, macOS and Linux.

First run: creates a private Python environment (.venv) and installs the
packages in requirements.txt. Every later run starts straight away.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / ".requirements.sha256"
REQUIREMENTS = ROOT / "requirements.txt"


def die(message):
    print()
    print("  " + message.replace("\n", "\n  "))
    print()
    sys.exit(1)


def venv_python():
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def venv_works():
    py = venv_python()
    if not py.exists():
        return False
    try:
        return subprocess.call([str(py), "-c", "import sys"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def main():
    if sys.version_info < (3, 9):
        die("Python 3.9 or newer is needed. Get it from https://www.python.org/downloads/")

    os.chdir(ROOT)

    # A .venv copied from another computer will not run here, so rebuild it.
    if VENV.exists() and not venv_works():
        shutil.rmtree(VENV, ignore_errors=True)

    if not venv_works():
        print("  First-time setup: creating a private Python environment ...")
        try:
            venv.EnvBuilder(with_pip=True).create(VENV)
        except Exception as exc:
            shutil.rmtree(VENV, ignore_errors=True)
            die("Could not create the Python environment:\n%s\n\n"
                "On Ubuntu/Debian, run:  sudo apt install python3-venv python3-pip" % exc)

    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not MARKER.exists() or MARKER.read_text().strip() != digest:
        print("  Installing required packages (a few minutes the first time) ...")
        result = subprocess.call([str(venv_python()), "-m", "pip", "install",
                                  "--disable-pip-version-check", "-r", str(REQUIREMENTS)])
        if result != 0:
            die("Could not install the required packages.\n"
                "Connect to the internet and run this launcher again.")
        MARKER.write_text(digest)

    try:
        sys.exit(subprocess.call([str(venv_python()), "app.py"]))
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
