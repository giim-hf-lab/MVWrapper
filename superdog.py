# -*- coding: utf-8 -*-
__all__ = []

import argparse
import os.path
import sys
from pathlib import Path


def _main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b", "--install-base",
        default=os.environ["PROGRAMFILES(X86)"],
        type=Path,
        dest="install_base"
    )
    parser.add_argument(
        "-v", "--version",
        default="2.5",
        dest="version"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=os.getcwd(),
        type=Path,
        dest="output_dir"
    )
    parser.add_argument(
        "-L", "--label",
        default="DEMOMA",
        dest="label"
    )

    parsed = parser.parse_args()

    with open(
        parsed.install_base / "Gemalto" / "SuperDog" / parsed.version / "VendorCodes" / f"{parsed.label}.hvc",
        "rb"
    ) as src, open(parsed.output_dir / f"superdog.vc", "wb") as dest:
        dest.write(b"static constexpr const char _SUPERDOG_VENDOR_CODE[] = R\"\"\"(")
        dest.write(src.read())
        dest.write(b")\"\"\";")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
