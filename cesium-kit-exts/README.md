# Cesium Kit Extensions

## Prerequisites

- Install Nvidia Omniverse: https://www.nvidia.com/en-us/omniverse/download/.
  - Use the default install location.
- Install Omniverse Code
- Install Connect Sample 200.0.0 (this exact version) and run `built.bat` (Windows) or `build.sh` (Linux) in the installation directory.
- Build the C++ library and install the Python bindings. For more detailed instructions see [Cesium for Omniverse](../cesium-omniverse/README.md).
    ```sh
    # Configure
    cmake -B build
    # Build
    cmake --build build --config Release
    # Install Python bindings
    cmake --install build --config Release --prefix ../cesium-kit-exts/exts/cesium.omniverse/cesium/omniverse/bindings/cesium/omniverse/ --component kit
    ```
- Open the VS Code workspace for your OS and install the recommended extensions. Make sure to open the workspace instead of opening the `cesium-kit-exts` folder directly, otherwise IntelliSense my not work properly.
  - [cesium-omniverse-linux.code-workspace](./.vscode/cesium-omniverse-linux.code-workspace)
  - [cesium-omniverse-windows.code-workspace](./.vscode/cesium-omniverse-windows.code-workspace)
- Run `link_app`. This will create a folder called `app` that is a symlink to the Omniverse Code installation folder. This allows VS Code Python IntelliSense to find the Omniverse Python libraries and select the same Python interpreter as Omniverse Code.
    ```sh
    # Windows
    link_app.bat --app code

    #Linux
    link_app.sh --app code
    ```
- Launch Omniverse Code. Add `exts` to the extension search paths so it can find our extensions. Then look for "cesium.omniverse" in the extension manager and enable it. Click the auto-load checkbox to load our extension on startup.

Extension Search Paths | Enable Extension
--|--
![Extension Search Paths](./images/extension-search-paths.png)|![Enable Extension](./images/enable-extension.png)

- Alternatively, Omniverse Code can be launched from the command line with our extension enabled.
    ```sh
    # Windows
    app/omni.code.bat --ext-folder exts --enable cesium.omniverse

    # Linux
    app/omni.code.sh --ext-folder exts --enable cesium.omniverse
    ```
