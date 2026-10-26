#!/usr/bin/env python3

import json
import os
import sys
import platform
import argparse
from typing import List

# List of libraries for which to copy the vcpkg description to the "notes" field in ThirdParty.json.
libs_with_notes = ["proj-data"]


def read_third_party_extra_json():
    path = "ThirdParty.extra.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def find_license(third_party_json, name):
    return next((entry for entry in third_party_json if entry["name"] == name), None)


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Create third party license json from vcpkg spdx files and ThirdParty.extra.json."
    )

    parser.add_argument(
        "--build-dir",
        help="The CMake build directory. From CMake variable PROJECT_BINARY_DIR.",
        action="store",
        required=True,
    )

    parser.add_argument(
        "--skip",
        help="Comma separated list of libraries to skip.",
        action="store",
    )

    parser.add_argument("--triplet", help="The target vcpkg triplet.", action="store")

    parser.add_argument(
        "--keep-description",
        help='Comma separated list of libraries for which to keep the vcpkg description (written to "notes" field).',
        action="store",
    )

    return parser.parse_args(argv)


def generate_third_party(argv: List[str]):
    args = parse_args(argv)
    skip_libs = args.skip.split(",") if args.skip is not None else []
    notes_libs = args.keep_description.split(",") if args.keep_description is not None else []
    build_dir = args.build_dir
    triplet = args.triplet
    third_party_extra_json = read_third_party_extra_json()
    third_party_json = []

    dep_dir = ""
    license_file = "vcpkg.spdx.json"
    current_platform = platform.system().lower()

    if triplet is None:
        if "linux" in current_platform:
            triplet = "x64-linux"
        elif "win" in current_platform:
            triplet = "x64-windows-static-md"

    dep_dir = os.path.join(build_dir, "vcpkg_installed", triplet, "share")

    for entry in os.listdir(dep_dir):
        dep_path = os.path.join(dep_dir, entry)
        if os.path.isdir(dep_path):
            license_path = os.path.join(dep_path, license_file)
            try:
                with open(license_path, "r", encoding="utf8") as f:
                    license_json = json.load(f)
                for p in license_json["packages"]:
                    if p["name"].lower() == entry.lower():  # There's other stuff in there we don't care about
                        info = {
                            "name": p["name"],
                            "license": [p["licenseConcluded"] if p["licenseConcluded"] != "NOASSERTION" else ""],
                            "url": p["homepage"] if "homepage" in p else "",
                            "version": p["versionInfo"],
                        }

                        # Write vcpkg description, if requested.
                        if p["name"] in notes_libs:
                            info["notes"] = p["description"]

                        if p["name"] not in skip_libs:
                            override = find_license(third_party_extra_json, p["name"]) or {}
                            if "license" in override:
                                info["license"] = override["license"]
                            if "notes" in override and p["name"] in notes_libs:
                                info["notes"] = override["notes"]
                            if "url" in override:
                                info["url"] = override["url"]
                            third_party_json.append(info)
            except Exception:
                pass

    for entry in third_party_extra_json:
        if "notes" in entry and not entry["name"] in notes_libs:
            del entry["notes"]

        if find_license(third_party_json, entry["name"]) is None:
            third_party_json.append(entry)

    third_party_json.sort(key=lambda x: x["name"].lower())

    with open("ThirdParty.json", "w", newline="\n") as f:
        json.dump(third_party_json, f, indent=4)


if __name__ == "__main__":
    generate_third_party(sys.argv[1:])
