#!/usr/bin/env python 

import os
import shutil
import subprocess

project_root = os.path.abspath(os.path.dirname(__file__))
schema_input_path = os.path.join(project_root, "exts/cesium.usd.plugins/schemas/cesium_schemas.usda")
schema_output_path = os.path.join(project_root, "src/plugins/CesiumUsdSchemas/src/CesiumUsdSchemas")
nvidia_usd_bins = os.path.join(project_root, "extern/nvidia/_build/target-deps/usd/release/bin")
nvidia_python_bins = os.path.join(project_root, "extern/nvidia/_build/target-deps/python")
nvidia_python_executable = os.path.join(nvidia_python_bins, "python.exe")
nvidia_python_libs = os.path.join(project_root, "extern/nvidia/_build/target-deps/python/lib")
nvidia_usd_python_libs = os.path.join(project_root, "extern/nvidia/_build/target-deps/usd/release/lib/python")
nvidia_usd_gen_schema = os.path.join(nvidia_usd_bins, "usdGenSchema")
os.environ["PYTHONPATH"] = f"{nvidia_usd_python_libs};{os.environ.get('PYTHONPATH', '')}"
os.environ["PATH"] = f"{nvidia_usd_bins};{nvidia_python_libs};{nvidia_python_bins};{nvidia_usd_python_libs};{os.environ.get('PATH', '')}"

subprocess.run([nvidia_python_executable, "-m", "pip", "install", "jinja2"])

# XXX: This doesn't work right. Not sure what is wrong.
# Temporarily move module.cpp and moduleDeps.cpp out of the folder.
shutil.move(os.path.join(schema_output_path, "module.cpp"), os.path.join(project_root, "module.cpp"))
shutil.move(os.path.join(schema_output_path, "moduleDeps.cpp"), os.path.join(project_root, "moduleDeps.cpp"))

# Clear out the old files.
for file in os.listdir(schema_output_path):
    os.remove(os.path.join(schema_output_path, file))

# Move module.cpp and moduleDeps.cpp back.
shutil.move(os.path.join(project_root, "module.cpp"), os.path.join(schema_output_path, "module.cpp"))
shutil.move(os.path.join(project_root, "moduleDeps.cpp"), os.path.join(schema_output_path, "moduleDeps.cpp"))

# Generate the new files.
subprocess.run([nvidia_python_executable, nvidia_usd_gen_schema, schema_input_path, schema_output_path])

# Move the generatedSchema.usda and plugInfo.json files up.
shutil.move(os.path.join(schema_output_path, "generatedSchema.usda"), os.path.join(schema_output_path, "../../generatedSchema.usda.in"))
shutil.move(os.path.join(schema_output_path, "plugInfo.json"), os.path.join(schema_output_path, "../../plugInfo.json.in"))

# Delete the Pixar junk from the first 23 lines of the generated code.
for file in os.listdir(schema_output_path):
    file_path = os.path.join(schema_output_path, file)
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path, "w") as f:
        f.writelines(lines[23:])
