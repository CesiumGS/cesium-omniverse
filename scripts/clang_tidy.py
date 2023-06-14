#!/usr/bin/env python3
import argparse
import sys
import shutil
from utils import utils
from typing import List


def parse_args(av: List[str]):
    parser = argparse.ArgumentParser(description="Run / check clang-tidy on staged cpp files.")
    parser.add_argument(
        "--clang-tidy-executable", help="Specific clang-tidy binary to use.", action="store", required=False
    )

    return parser.parse_known_args(av)


def main(av: List[str]):
    known_args, clang_tidy_args = parse_args(av)
    project_root = utils.get_project_root()

    clang_tidy_executable = known_args.clang_tidy_executable
    if not clang_tidy_executable:
        clang_tidy_executable = shutil.which("clang-tidy")

    project_root = utils.get_project_root()
    candidate_files = [
        f.as_posix() for f in utils.get_staged_git_files(project_root) if f.suffix in utils.CPP_EXTENSIONS
    ]

    cmd = [clang_tidy_executable] + clang_tidy_args + candidate_files
    if len(candidate_files) > 0:
        print("Running clang-tidy")
        utils.run_command_and_echo_on_error(cmd)
    else:
        print("Skipping clang-tidy (no cpp files staged)")


if __name__ == "__main__":
    main(sys.argv[1:])
