"""
This file is a post build step run by cmake that copies over the README.md, CHANGES.md, and related resources to
the exts/docs folder for packaging.
"""
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class PathPair:
    """
    Represents a source and relative destination pair.

    :arg source: The source path for the file.
    :arg relative_destination: The relative destination for the file.
    """
    source: Path
    relative_destination: str = ""


def find_resources(path: Path) -> List[PathPair]:
    """
    Finds all resources within a file and returns them as a list of PathPairs. The search is done using a regular
    expression looking for all links that contain the substring "docs/resources".

    NOTE: This **only** works with relative paths. Absolute paths in the file read will fail.

    :param path: The file to search.
    :return: A list containing PathPairs of all resources found in the file.
    """
    regex = re.compile(r"!\[.*]\((.*docs/(resources.*?))\)")
    root_path = path.parent.resolve()

    resources: List[PathPair] = []

    with open(path.resolve(), "r") as f:
        for line in f.readlines():
            match = regex.search(line)
            if match is not None:
                source = root_path.joinpath(match.group(1))
                relative_destination = match.group(2)
                resources.append(PathPair(source, relative_destination))

    return resources


def copy_to_destination(pair: PathPair, destination: Path) -> None:
    """
    Copies the file based on the path and relative destination contained in the pair.

    NOTE: This uses shutils so if you're on a version of Python older than 3.8 this will be slow.

    :param pair: The PathPair for the copy operation.
    :param destination: The path of the destination directory.
    """
    true_destination = destination.joinpath(
        pair.relative_destination) if pair.relative_destination != "" else destination

    # In the event that true_destination isn't a direct file path, we need to take the source filename and append it
    # to true_destination.
    if true_destination.is_dir():
        true_destination = true_destination.joinpath(pair.source.name)

    true_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pair.source, true_destination)


def main() -> int:
    project_root = Path(__file__).parent.parent
    destination = project_root.joinpath("exts/cesium.omniverse/doc")

    # This is separated out because we need it for finding resources.
    readme_path = project_root.joinpath("README.md")

    try:
        # Turning off formatting here for readability.
        # fmt: off
        paths_to_copy: List[PathPair] = [
            PathPair(readme_path),
            PathPair(project_root.joinpath("CHANGES.md")),
            *find_resources(readme_path)
        ]
        # fmt: on

        for pair in paths_to_copy:
            copy_to_destination(pair, destination)
    except Exception as e:
        print(e)
        return 1

    return 0


exit(main())
