import os
import argparse

import repoman
import packmanapi


def run_command():
    repo_folders = repoman.api.get_repo_paths()

    # Fetch the asset dependencies
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--platform-target',
                        dest='platform_target', required=False)
    options, _ = parser.parse_known_args()

    # Checking if platform was passed
    # We cannot use argparse's default, as we also need to set up the command line argument
    # if it wasn't supplied. It is possible to also check for the host platform, if we want to
    # make different default behavior when building on windows.
    if not repoman.api.has_options_arg(options, 'platform_target'):
        options.platform_target = 'linux-x86_64'

    deps_target = [repo_folders["target_deps_xml"], "deps/usd-deps.packman.xml"]
    repo_root = repo_folders["root"]

    # Dependencies will be fetched from the "deps" folder files
    for deps_target in deps_target:
        packmanapi.pull(os.path.join(repo_root, deps_target), platform=options.platform_target)


if __name__ == "__main__" or __name__ == "__mp_main__":
    run_command()
