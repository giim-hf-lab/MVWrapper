# -*- coding: utf-8 -*-
__all__ = []

import argparse
import sys
from pathlib import Path


def _main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        dest="input_file"
    )
    parser.add_argument(
        "--name",
        required=True,
        dest="name"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        dest="output_file"
    )

    parsed = parser.parse_args()

    with open(parsed.input_file, "r") as rf, open(parsed.output_file, "w") as wf:
        wf.write(f"static constexpr const char _{parsed.name}_VENDOR_CODE[] = R\"\"\"({rf.read()})\"\"\";")

    return 0


if __name__ == "__main__":
    sys.exit(_main())
