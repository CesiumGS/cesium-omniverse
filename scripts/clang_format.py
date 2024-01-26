#!/usr/bin/env python3
import argparse
import sys
import subprocess
import shutil
import shlex
from utils import utils
from pathlib import Path
from typing import List


def clang_format_on_path(clang_format_binary: str, absolute_path: Path) -> str:
    cmd = "{} -style=file {}".format(shlex.quote(clang_format_binary), shlex.quote(str(absolute_path)))
    cmd = shlex.split(cmd)
    result = subprocess.check_output(cmd)
    return result.decode("utf-8", "replace")


def clang_format_in_place(clang_format_binary: str, absolute_path: Path):
    cmd = "{} -style=file -i {}".format(shlex.quote(clang_format_binary), shlex.quote(str(absolute_path)))
    cmd = shlex.split(cmd)
    subprocess.check_output(cmd)


def parse_args(av: List[str]):
    parser = argparse.ArgumentParser(description="Run / check clang-formatting.")
    parser.add_argument(
        "--clang-format-executable", help="Specific clang-format binary to use.", action="store", required=False
    )
    parser.add_argument(
        "--source-directories",
        help='Directories (relative to project root) to recursively scan for cpp files (e.g "src", "include"...)',
        nargs="+",
        required=True,
    )
    run_type = parser.add_mutually_exclusive_group(required=True)
    run_type.add_argument(
        "--fix", help="Apply clang-format formatting to source in-place (destructive)", action="store_true"
    )
    run_type.add_argument("--check", help="Check if source matches clang-format rules", action="store_true")
    scope_type = parser.add_mutually_exclusive_group(required=True)
    scope_type.add_argument("--all", help="Process all valid source files.", action="store_true")
    scope_type.add_argument("--staged", help="Process only staged source files.", action="store_true")
    return parser.parse_args(av)


def main(av: List[str]):
    if not shutil.which("git"):
        raise RuntimeError("Could not find git in path")

    project_root_directory = utils.get_project_root()
    args = parse_args(av)

    # Use user provided clang_format binary if provided
    clang_format_binary = args.clang_format_executable
    if clang_format_binary:
        clang_format_binary = shutil.which(clang_format_binary)
    if not clang_format_binary:
        clang_format_binary = shutil.which("clang-format")
    if not clang_format_binary:
        raise RuntimeError("Could not find clang-format in system path")

    mode = "all" if args.all else "staged"
    source_directories = args.source_directories

    # Generate list of source_files to check / fix.
    source_files: List[utils.SourceFile] = utils.get_source_files(source_directories, args.all)
    failed_files: List[utils.FailedFile] = []

    # Fix or check formatting for each file
    for src in source_files:
        absolute_path = project_root_directory.joinpath(src.relative_path)
        if args.check:
            old_text = (
                absolute_path.read_text(encoding="utf-8")
                if not src.staged
                else utils.get_staged_file_text(src.relative_path)
            )
            new_text = clang_format_on_path(clang_format_binary, absolute_path)
            diff = utils.unidiff_output(old_text, new_text)
            if diff != "":
                failed_files.append(utils.FailedFile(src.relative_path, diff))
        else:
            clang_format_in_place(clang_format_binary, absolute_path)

    if len(source_files) == 0:
        print("clang-format ({} files): No files found, nothing to do.".format(mode))
        sys.exit(0)

    if args.fix:
        print("Ran clang-format -style=file -i on {} files".format(mode))
        sys.exit(0)

    if len(failed_files) == 0:
        print("clang-format ({} files) passes.".format(mode))
        sys.exit(0)

    print("clang-format ({} files) failed on the following files: ".format(mode))
    for failure in failed_files:
        print("{}".format(failure.relative_path))
        print(failure.diff)

    sys.exit(len(failed_files))


if __name__ == "__main__":
    main(sys.argv[1:])
