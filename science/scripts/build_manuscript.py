"""Platform-neutral Tectonic manuscript build entrypoint."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manuscript = root / "manuscript"
    tectonic = shutil.which("tectonic")
    latexmk = shutil.which("latexmk")
    if tectonic is not None:
        command = [tectonic, "main.tex", "--keep-logs", "--keep-intermediates"]
    elif latexmk is not None:
        command = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    else:
        print("Install Tectonic 0.17+ or latexmk with pdfLaTeX.", file=sys.stderr)
        return 2
    completed = subprocess.run(command, cwd=manuscript, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
