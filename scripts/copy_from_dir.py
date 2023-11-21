import sys
from pathlib import Path
from shutil import copy2

# Broken out for formatting reasons, since tabs within HEREDOCs will be output.
usage_message = """Invalid arguments.
    Usage: copy_from_dir.py <glob-pattern> <source-dir-path> <destination-dir-path>

Please fix your command and try again.
"""


def main():
    if len(sys.argv) < 4:
        print(usage_message)
        return 1

    glob_pattern: str = sys.argv[1]
    source_dir = Path(sys.argv[2]).resolve()
    dest_dir = Path(sys.argv[3]).resolve()

    print(f'Performing file copy with glob pattern "{glob_pattern}"')
    print(f"\tSource: {source_dir}")
    print(f"\tDestination: {dest_dir}\n")

    source_files = source_dir.glob(glob_pattern)

    for f in source_files:
        source_path = source_dir / f

        copy2(source_path, dest_dir, follow_symlinks=True)

        print(f"Copied {source_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
