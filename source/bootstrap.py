# Runtime bootstrap for KhunDech.
# This file keeps deployment simple:
# 1) Check whether required dependencies are importable.
# 2) Install from requirements.txt only when something is missing.
# 3) Start the bot with the current Python interpreter.
#
# This file is part of KhunDech.
# KhunDech is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path


REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")

# Some pip package names differ from import module names.
IMPORT_NAME_MAP = {
    "discord.py": "discord",
    "google-genai": "google.genai",
}


def _parse_requirement_names() -> list[str]:
    # Return top-level package names from requirements.txt.
    if not REQUIREMENTS_FILE.exists():
        return []

    package_names: list[str] = []
    for raw_line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Skip pip options and editable/path installs.
        if line.startswith(("-", ".", "/")):
            continue

        # Extract package token before any version/operator marker.
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if not match:
            continue

        package_names.append(match.group(1))

    return package_names


def _missing_imports(package_names: list[str]) -> list[str]:
    # Return package names that are not currently importable.
    missing: list[str] = []
    for package_name in package_names:
        import_name = IMPORT_NAME_MAP.get(package_name, package_name.replace("-", "_"))
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(package_name)
    return missing


def _install_requirements() -> None:
    # Install dependencies from requirements.txt.
    if not REQUIREMENTS_FILE.exists():
        print("[bootstrap] requirements.txt not found, skipping dependency install.")
        return

    print("[bootstrap] Installing dependencies from requirements.txt...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(REQUIREMENTS_FILE)],
        check=True,
    )


def main() -> None:
    package_names = _parse_requirement_names()
    missing = _missing_imports(package_names)

    if missing:
        print(f"[bootstrap] Missing packages detected: {', '.join(missing)}")
        _install_requirements()
    else:
        print("[bootstrap] All required packages are available.")

    print("[bootstrap] Starting bot...")
    subprocess.run([sys.executable, "-u", "bot.py"], check=True)


if __name__ == "__main__":
    main()
