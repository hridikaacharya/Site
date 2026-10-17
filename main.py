#!/usr/bin/env python3
"""
List all filenames currently in the _Keep folder into a plain text file.

USAGE
    python main.py --source "/path/to/footage"

This creates keep_list.txt inside SOURCE_FOLDER, one filename per line,
sorted alphabetically. Doesn't touch or move any files.
"""

import os
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="List filenames in _Keep folder.")
    parser.add_argument("--source", required=True, help="The same source folder you passed to media_culler.py")
    parser.add_argument("--output", default=None, help="Output file name (default: keep_list.txt inside source folder)")
    args = parser.parse_args()

    keep_dir = Path(args.source) / "_Keep"
    if not keep_dir.is_dir():
        sys.exit(f"No _Keep folder found at {keep_dir}. Did you run media_culler.py --mode sort first?")

    files = sorted(f.name for f in keep_dir.iterdir() if f.is_file())

    output_path = Path(args.output) if args.output else Path(args.source) / "keep_list.txt"
    with open(output_path, "w") as f:
        for name in files:
            f.write(name + "\n")

    print(f"{len(files)} filenames written to {output_path}")


if __name__ == "__main__":
    main()