import os
import argparse

import repoman
import packmanapi


def run_command():
    repo_folders = repoman.api.get_repo_paths()

    # Fetch the asset dependencies
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform-target", dest="platform_target", required=False)
    options, _ = parser.parse_known_args()

    # Checking if platform was passed
    # We cannot use argparse's default, as we also need to set up the command line argument
    # if it wasn't supplied. It is possible to also check for the host platform, if we want to
    # make different default behavior when building on windows.
    if not repoman.api.has_options_arg(options, "platform_target"):
        options.platform_target = "linux-x86_64"

    deps_folder = repo_folders["deps_xml_folder"]
    deps_targets = [
        "kit-sdk.packman.xml",
        "target-deps.packman.xml",
    ]

    # Dependencies will be fetched from the "deps" folder files
    for deps_target in deps_targets:
        packmanapi.pull(os.path.join(deps_folder, deps_target), platform=options.platform_target)


if __name__ == "__main__" or __name__ == "__mp_main__":
    run_command()
