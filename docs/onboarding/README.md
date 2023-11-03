## What is Omniverse?
Omniverse is a tool that provides an interface for various other tools to interact with a shared 3d environment. The core of this is a USD stage and a Fabric stage. The tools that interact with these stages do so via extensions.

To better understand extensions and how they're defined, check out the [official Omniverse extension template](https://github.com/NVIDIA-Omniverse/kit-extension-template) for a "hello world" extension.
There is also a similar [C++ extension template](https://github.com/NVIDIA-Omniverse/kit-extension-template-cpp).

### Our Extensions/Apps
- Cesium for Omniverse ("The main extension")
    - Responsible for streaming geospatial data onto the stages, and providing the user interface.
- Cesium Usd plugins
    - Required by the main extension to facilitatge interactions with the USD stage.
- Cesium Powertools
    - Helpful additions for developers, such as one-click ways to open debug interfaces and print the fabric stage.
- Cesium Cpp Tests
    - Tests of the C++ code underlying the main extension. For more info see [the testing guide](../testing-guide/README.md)
- The Performance App
    - Used to get general timing of an interactive session. See [the testing guide](../testing-guide/README.md) for how to run.

## Project File Structure
Some self-explanatory directories have been ommitted.

- `apps` - Tools that use the extensions, such as the performance testing app, but are not themselves extensions
- `docker` - Docker configuration for AlmaLinux 8 CI builds
- `exts` - This is where extension code is kept. The file structure follows the pattern:
    ```
    exts
    └── dot.separated.name
        ├── bin
        │   ├── libdot.separated.name.plugin.so
        └── dot
            └── separated
                └── name
                    └── codeNeededByExtension
                    └── __init__.py
                    └── extension.py
    ```
- `genStubs.*`-  auto-generates stub files for python bindings, which are not functionally required but greatly improve intellisense.
- `src`/`include` - There are several `src`/`include` subdirs throughout the project, but this top level one is only for code used in the python bindings for the main extension.
- `regenerate_schema.*` - changes to our usd schema require using this script.
- `scripts` - useful scripts for development that do not contribute to any extension function.
- `tests` - c++ related test code used by the Tests Extension. For python related test code, check `exts/cesium.omniverse/cesium/omniverse/tests`. For more details, see the [testing guide](../testing-guide/README.md)
