from os import getcwd
from os.path import exists
import json
from jsmin import jsmin
# import subprocess

#
# Note: This requires JSMin to be installed since the vscode workspace files have Comments in them.
#   You can install JSMin by just installing it globally via pip.
#
# Also Note: You may need to run Visual Studio and open a Python file before running this script.
#
cwd_path = getcwd()
root_path = f"{cwd_path}/../extern/nvidia/app"
vs_code_workspace_file_path = f"{cwd_path}/../.vscode/cesium-omniverse-linux.code-workspace"
output_path = f"{cwd_path}/python_path.txt"

if not exists(root_path):
    print(f"Could not find {root_path}")
    exit(1)

if not exists(vs_code_workspace_file_path):
    print(f"Could not find {vs_code_workspace_file_path}")
    exit(1)

print(f"Using root path: {root_path}")
print(f"Using vs code workspace file: {vs_code_workspace_file_path}")

with open(vs_code_workspace_file_path) as fh:
    m = jsmin(fh.read())
    vs_code_workspace_file = json.loads(m)


def process_paths(path):
    return path.replace("${workspaceFolder}", cwd_path)


extra_paths = list(map(process_paths, vs_code_workspace_file["settings"]["python.analysis.extraPaths"]))

export_python_path_command = "export PYTHONPATH=\""

for path in extra_paths:
    export_python_path_command = f"{export_python_path_command}{path}:"

export_python_path_command = export_python_path_command + "$PYTHONPATH\""

print(f"Copy this to your .zshrc or other appropriate file:\n\n{export_python_path_command}\n")

with open(output_path, 'w') as fh:
    fh.write(export_python_path_command)

print(f"Because we aren't sadists, this has also been dumped to a file located at {output_path}\n\n")

# TODO: Figure out why this isn't working.
# apply_to_session = input("Do you want to apply this to your current terminal session? [y\\N]").lower() in ["y", "yes"]
#
# if apply_to_session:
#     subprocess.run(export_python_path_command)
#     print("Applied to terminal.")
