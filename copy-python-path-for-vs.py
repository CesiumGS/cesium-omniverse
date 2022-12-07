from os import getcwd
from os.path import exists
import json
from textwrap import indent
from jsmin import jsmin

#
# Note: This requires JSMin to be installed since the vscode workspace files have Comments in them.
#   You can install JSMin by just installing it globally via pip.
#
# Also Note: You may need to run Visual Studio and open a Python file before running this script.
#
cwd_path = getcwd()
root_path = f"{cwd_path}/extern/nvidia/app"
vs_code_workspace_file_path = f"{cwd_path}/.vscode/cesium-omniverse-windows.code-workspace"
vs_python_settings_file_path = f"{cwd_path}/.vs/PythonSettings.json"

if not exists(root_path):
    print(f"Could not find {root_path}")
    exit(1)

if not exists(vs_code_workspace_file_path):
    print(f"Could not find {vs_code_workspace_file_path}")
    exit(1)

if not exists(vs_python_settings_file_path):
    print(f"Could not find {vs_python_settings_file_path}")
    exit(1)

print(f"Using root path: {root_path}")
print(f"Using vs code workspace file: {vs_code_workspace_file_path}")
print(f"Using vs PythonSettings file: {vs_python_settings_file_path}")

with open(vs_code_workspace_file_path) as fh:
    m = jsmin(fh.read())
    vs_code_workspace_file = json.loads(m)

def process_paths(path):
    return path.replace("${workspaceFolder}", cwd_path).replace("/", "\\")

extra_paths = list(map(process_paths, vs_code_workspace_file["settings"]["python.analysis.extraPaths"]))

with open(vs_python_settings_file_path, 'r') as fh:
    vs_python_settings = json.load(fh)

vs_python_settings["SearchPaths"] = extra_paths

# The read and write handles are split because we want to truncate the old file.
with open(vs_python_settings_file_path, 'w') as fh:
    json.dump(vs_python_settings, fh, indent=2)

print(f"Wrote to {vs_python_settings_file_path}")
