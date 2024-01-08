#!/usr/bin/env python3
import sys
import subprocess
import multiprocessing
import os
import platform
import shutil

try:
    import pty
except Exception:
    pass
import webbrowser
from typing import List, NamedTuple


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def process(cmd: List[str]):
    print("Run: " + " ".join(cmd))

    if is_linux():
        # Using pty instead of subprocess to get terminal colors
        result = pty.spawn(cmd)
        if result != 0:
            sys.exit(result)
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
        p.communicate()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, p.args)


def open_browser(html: str):
    html = os.path.realpath(html)
    html = "file://{}".format(html)
    webbrowser.open(html, new=2)


class Args(NamedTuple):
    task: str
    build_folder: str
    build_type: str
    tracing: bool
    verbose: bool
    kit_debug: bool
    parallel: bool
    build_only: bool


def get_cmake_configure_command(args: Args):
    cmd = ["cmake", "-B", args.build_folder]

    cmd.extend(("-D", "CMAKE_BUILD_TYPE={}".format(args.build_type)))

    if args.tracing:
        cmd.extend(("-D", "CESIUM_OMNI_ENABLE_TRACING=ON"))

    if args.kit_debug:
        cmd.extend(("-D", "CESIUM_OMNI_USE_NVIDIA_DEBUG_LIBRARIES=ON"))

    if is_windows():
        cmd.extend(("-G", "Ninja Multi-Config", "-D", "CMAKE_C_COMPILER=cl", "-D", "CMAKE_CXX_COMPILER=cl"))
        return cmd

    return cmd


def get_cmake_build_command(args: Args, target: str):
    cmd = ["cmake", "--build", args.build_folder]

    if is_windows():
        cmd.extend(("--config", args.build_type))

    if target:
        cmd.extend(("--target", target))

    if args.verbose:
        cmd.append("--verbose")

    if args.parallel:
        # use every core except one so that computer doesn't go too slow
        cores = max(1, multiprocessing.cpu_count() - 1)
        cmd.extend(("--parallel", str(cores)))

    return cmd


def get_cmake_install_command(args: Args):
    cmd = ["cmake", "--install", args.build_folder]

    if is_windows():
        cmd.extend(("--config", args.build_type))

    return cmd


def configure(args: Args):
    configure_cmd = get_cmake_configure_command(args)
    process(configure_cmd)


def build(args: Args):
    build_cmd = get_cmake_build_command(args, None)
    install_kit_cmd = get_cmake_install_command(args)

    if not args.build_only:
        configure_cmd = get_cmake_configure_command(args)
        process(configure_cmd)

    process(build_cmd)
    process(install_kit_cmd)


def coverage(args: Args):
    if is_windows():
        print("Coverage is not supported for Windows")
        return

    configure_cmd = get_cmake_configure_command(args)
    build_cmd = get_cmake_build_command(args, "generate-coverage")
    html = "{}/coverage/index.html".format(args.build_folder)

    process(configure_cmd)
    process(build_cmd)
    open_browser(html)


def documentation(args: Args):
    configure_cmd = get_cmake_configure_command(args)
    documentation_cmd = get_cmake_build_command(args, "generate-documentation")
    html = "{}/docs/html/index.html".format(args.build_folder)

    process(configure_cmd)
    process(documentation_cmd)
    open_browser(html)


def install(args: Args):
    configure_cmd = get_cmake_configure_command(args)
    install_cmd = get_cmake_build_command(args, "install")
    process(configure_cmd)
    process(install_cmd)


def clean(args: Args):
    if os.path.exists(args.build_folder) and os.path.isdir(args.build_folder):
        shutil.rmtree(args.build_folder)


def format(args: Args):
    format_cmd = get_cmake_build_command(args, "clang-format-fix-all")
    process(format_cmd)


def lint(args: Args):
    clang_tidy_cmd = get_cmake_build_command(args, "clang-tidy")
    process(clang_tidy_cmd)


def lint_fix(args: Args):
    clang_tidy_cmd = get_cmake_build_command(args, "clang-tidy-fix")
    process(clang_tidy_cmd)


def dependency_graph(args: Args):
    configure_cmd = get_cmake_configure_command(args)
    conan_packages_path = os.path.join(args.build_folder, "Conan_Packages")
    dependency_html = os.path.join(args.build_folder, "dependency_graph.html")
    dependency_cmd = ["conan", "info", args.build_folder, "-if", conan_packages_path, "--graph", dependency_html]

    process(configure_cmd)
    process(dependency_cmd)
    open_browser(dependency_html)


def get_build_folder_name(build_type: str):
    folder_name = "build"

    if is_windows():
        return folder_name

    if build_type != "Release":
        folder_name += "-{}".format(build_type.lower())

    return folder_name


def get_bin_folder_name(build_type: str):
    build_folder_name = get_build_folder_name(build_type)

    if is_windows():
        bin_folder_name = "{}/bin/{}".format(build_folder_name, build_type)
    else:
        bin_folder_name = "{}/bin".format(build_folder_name)

    return bin_folder_name


def main(av: List[str]):
    print(av)
    task = av[0]
    build_type = av[1] if len(av) >= 2 else "Release"
    build_folder = get_build_folder_name(build_type)
    tracing = True if len(av) >= 3 and av[2] == "--tracing" else False
    verbose = True if len(av) >= 3 and av[2] == "--verbose" else False
    kit_debug = True if len(av) >= 3 and av[2] == "--kit-debug" else False
    parallel = False if len(av) >= 4 and av[3] == "--no-parallel" else True
    build_only = True if len(av) >= 3 and av[2] == "--build-only" else False
    args = Args(task, build_folder, build_type, tracing, verbose, kit_debug, parallel, build_only)

    if task == "configure":
        configure(args)
    elif task == "build":
        build(args)
    elif task == "clean":
        clean(args)
    elif task == "coverage":
        coverage(args)
    elif task == "documentation":
        documentation(args)
    elif task == "install":
        install(args)
    elif task == "format":
        format(args)
    elif task == "lint":
        lint(args)
    elif task == "lint-fix":
        lint_fix(args)
    elif task == "dependency-graph":
        dependency_graph(args)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e)
        exit(1)
