#!/usr/bin/env python3
import subprocess
import shlex
import os
import glob
import sys
from pathlib import Path
from typing import List, NamedTuple, Set
import difflib

CPP_EXTENSIONS = [".cpp", ".h", ".cxx", ".hxx", ".hpp", ".cc", ".inl"]


def get_project_root() -> Path:
    try:
        cmd = shlex.split('git rev-parse --show-toplevel')
        output = subprocess.check_output(
            cmd).strip().decode('utf-8', 'replace')
        return Path(output)
    except subprocess.CalledProcessError:
        raise RuntimeError('command must be ran inside .git repo')


def get_staged_git_files(project_root: Path) -> List[Path]:
    cmd = shlex.split("git diff --cached --name-only --diff-filter=ACMRT")
    paths = subprocess.check_output(cmd).decode('utf-8').splitlines()
    return [project_root.joinpath(p) for p in paths]


def get_cmake_build_directory(project_root: Path):
    glob_pattern = project_root.joinpath("**/CMakeCache.txt").as_posix()
    results = glob.glob(glob_pattern, recursive=True)
    if len(results) == 0:
        err = "Could not find CMakeCache.txt in {}. Generate CMake configuration first.".format(
            project_root)
        raise RuntimeError(err)

    cmake_build_directory = os.path.realpath(
        os.path.join(project_root, results[0], ".."))
    return cmake_build_directory


def run_cmake_target(cmake_build_directory, target):
    path = shlex.quote(cmake_build_directory)
    cmd = shlex.split("cmake --build {} --target {}".format(path, target))
    run_command_and_echo_on_error(cmd)


def run_command_and_echo_on_error(cmd: List[str]):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Command \"{}\" failed:".format(' '.join(cmd)))
        print(e.output.decode('utf-8'))
        sys.exit(1)


class SourceFile(NamedTuple):
    relative_path: Path
    staged: bool


class FailedFile(NamedTuple):
    relative_path: Path
    diff: str


def get_source_files(source_directories: List[str], modeIsAll: bool) -> List[SourceFile]:
    project_root = get_project_root()
    staged_rel_paths = get_staged_rel_paths()
    source_files = []
    for directory in source_directories:
        for extension in CPP_EXTENSIONS:
            glob_pattern = os.path.join(
                project_root, directory, "**/*" + extension)
            glob_results = glob.glob(glob_pattern, recursive=True)
            for abs_path in glob_results:
                rel_path = Path(abs_path).relative_to(project_root)
                source_files.append(SourceFile(
                    rel_path, rel_path in staged_rel_paths))
    return list(filter(lambda source_file: source_file.staged or modeIsAll, source_files))


def get_staged_rel_paths() -> Set[str]:
    cmd = shlex.split("git diff --cached --name-only --diff-filter=ACMRT")
    staged_rel_paths = subprocess.check_output(cmd)
    staged_rel_paths = staged_rel_paths.decode('utf-8', 'replace')
    return set([Path(path) for path in staged_rel_paths.splitlines()])


def get_staged_file_text(relative_path: Path) -> str:
    cmd = "git show :{}".format(shlex.quote(str(relative_path.as_posix())))
    cmd = shlex.split(cmd)
    output = subprocess.check_output(cmd).decode('utf-8', 'replace')
    return output


COLOR_SUPPORT = False
try:
    import colorama
    colorama.init()
    COLOR_SUPPORT = True

    def color_diff(diff):
        for line in diff:
            if line.startswith('+'):
                yield colorama.Fore.GREEN + line + colorama.Fore.RESET
            elif line.startswith('-'):
                yield colorama.Fore.RED + line + colorama.Fore.RESET
            elif line.startswith('^'):
                yield colorama.Fore.BLUE + line + colorama.Fore.RESET
            else:
                yield line
except ImportError:
    pass


def unidiff_output(expected: str, actual: str):
    expected = expected.splitlines(1)
    actual = actual.splitlines(1)
    diff = difflib.unified_diff(expected, actual)
    if COLOR_SUPPORT:
        diff = color_diff(diff)
    return ''.join(diff)
