# -*- coding: utf-8 -*-
__all__ = []

import argparse
import sys
from pathlib import Path


def _main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b", "--install-base",
        type=Path,
        required=True,
        dest="install_base"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        dest="output_dir"
    )
    parser.add_argument(
        "-L", "--label",
        required=True,
        dest="label"
    )

    parsed = parser.parse_args()

    with open(
        parsed.install_base / "VendorCodes" / f"{parsed.label}.hvc",
        "rb"
    ) as src, open(parsed.output_dir / f"superdog.vc", "wb") as dest:
        dest.write(b"static constexpr const char _SUPERDOG_VENDOR_CODE[] = R\"\"\"(")
        dest.write(src.read())
        dest.write(b")\"\"\";")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
