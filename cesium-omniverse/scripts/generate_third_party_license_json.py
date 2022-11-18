#!/usr/bin/env python3
import json
import os
import shlex
import subprocess
import argparse
from typing import List
import sys
from pathlib import Path


def main(argv: List[str]):
    args = parse_args(argv)
    project_dir = args.project_dir
    build_dir = args.build_dir
    libraries_to_skip = args.skip.split(',')

    cmd = "conan info {} -if {} -j".format(build_dir,
                                           os.path.join(build_dir, 'Conan_Packages'))
    cmd = shlex.split(cmd, posix=(os.name == 'posix'))
    try:
        output = subprocess.check_output(cmd).decode('utf-8')
        json_output = output.split(os.linesep, 2)[1]
        third_party_json = json.loads(json_output)
    except subprocess.CalledProcessError as error:
        cmd_string = ' '.join(error.cmd)
        raise RuntimeError('Conan command \'{}\' failed with error {}. Third party JSON creation aborted.'
                           .format(cmd_string, error.returncode))

    third_party_json = generate_conan_third_party_json(
        third_party_json, libraries_to_skip)
    third_party_extra_json = json.loads(Path(project_dir).joinpath(
        'ThirdParty.extra.json').read_text())

    # Handle ThirdParty.extra.json
    for element in third_party_extra_json:
        if 'override' in element:
            found_match = False
            for match in third_party_json:
                if match['name'] == element['name']:
                    found_match = True
                    break
            if found_match:
                del element['override']
                third_party_json.remove(match)
                combined = {**match, **element}
                third_party_json.append(combined)
            else:
                raise RuntimeError('Could not find library to override: \'{}\'. Third party JSON creation aborted.'
                                   .format(element.name))
        else:
            third_party_json.append(element)

    third_party_json.sort(key=lambda obj: obj['name'].lower())
    third_party_json_path = os.path.join(project_dir, 'ThirdParty.json')

    with open(third_party_json_path, 'w', newline='\n') as json_file:
        json.dump(third_party_json, json_file, indent=4)
        json_file.write('\n')


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(
        description='Create third party license json from Conan info and ThirdParty.extra.json.'
    )

    parser.add_argument('--project-dir',
                        help='The project directory.',
                        action='store',
                        required='true'
                        )

    parser.add_argument('--build-dir',
                        help='The CMake build directory. From CMake variable PROJECT_BINARY_DIR.',
                        action='store',
                        required='true'
                        )

    parser.add_argument('--skip',
                        help='Comma separated list of libraries to skip.',
                        action='store',
                        )

    return parser.parse_args(argv)


def generate_conan_third_party_json(third_party_json, libraries_to_skip):
    result = []
    for library in third_party_json:
        # skip the `conanfile` object, as its NOT a real third party library
        if library['reference'] == 'conanfile.txt':
            continue

        display_name = library['display_name']
        url = library['homepage']
        license = library['license']

        licenses = []
        for l in license:
            licenses.extend(l.split(', '))

        display_name_pieces = display_name.split('/')
        name = display_name_pieces[0]
        version = display_name_pieces[1]

        # skip libraries that aren't included in the executable
        if name in libraries_to_skip:
            continue

        result.append({
            'name': name,
            'license': licenses,
            'version': version,
            'url': url
        })
    return result


if __name__ == '__main__':
    main(sys.argv[1:])
