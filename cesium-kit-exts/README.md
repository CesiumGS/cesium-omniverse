# Cesium Kit Extensions

## Prerequisites

- Install Nvidia Omniverse: https://www.nvidia.com/en-us/omniverse/download/
- Install Omniverse Create 2022.3.0 (or later)
- Go to the [`cesium-omniverse`](../cesium-omniverse/) directory and build the C++ project

    ```sh
    # Windows
    cmake -B build
    cmake --build build --config Release
    cmake --install build --config Release --component kit
    ```
    ```sh
    # Linux
    cmake -B build -D CMAKE_BUILD_TYPE=Release
    cmake --build build
    cmake --install build --component kit
    ```

- Launch Omniverse Create. Add `exts` to the extension search paths so it can find our extensions. Then look for "cesium.omniverse" in the extension manager and enable it. Click the auto-load checkbox to load our extension on startup.
  Extension Search Paths | Enable Extension
  --|--
  ![Extension Search Paths](./images/extension-search-paths.png)|![Enable Extension](./images/enable-extension.png)

- You should see a UI window appear. Click `Create Tileset` and then `Update Frame`

  ![Plugin](./images/plugin.png)


## VS Code development

- Run `link_app`. This will create a folder called `app` that is a symlink to the Omniverse Code installation folder. This allows VS Code Python IntelliSense to find the Omniverse Python libraries and select the same Python interpreter as Omniverse Code.
    ```sh
    # Windows
    link_app.bat --app create
    ```
    ```sh
    # Linux
    link_app.sh --app create
    ```
- Open the VS Code workspace for your OS and install the recommended extensions. Make sure to open the workspace instead of opening the `cesium-kit-exts` folder directly, otherwise IntelliSense my not work properly.
  - [cesium-omniverse-linux.code-workspace](./.vscode/cesium-omniverse-linux.code-workspace)
  - [cesium-omniverse-windows.code-workspace](./.vscode/cesium-omniverse-windows.code-workspace)
