<!-- omit in toc -->
# Cesium for Omniverse

- [Prerequisites](#prerequisites)
  - [Linux](#linux)
  - [Windows](#windows)
- [Clone the repository](#clone-the-repository)
- [Build](#build)
  - [Linux](#linux-1)
  - [Windows](#windows-1)
  - [Advanced build options](#advanced-build-options)
  - [CMake Options](#cmake-options)
- [Unit Tests](#unit-tests)
- [Coverage](#coverage)
- [Documentation](#documentation)
- [Installing](#installing)
- [Sanitizers](#sanitizers)
- [Formatting](#formatting)
- [Linting](#linting)
- [VSCode](#vscode)
  - [Workspaces](#workspaces)
  - [Debugging](#debugging)
  - [Tasks](#tasks)
- [Project Structure](#project-structure)
- [Third Party Libraries](#third-party-libraries)

## Prerequisites

See [Linux](#linux) or [Windows](#windows) for step-by-step installation instructions

- Linux (Ubuntu 22.04+ or equivalent) or Windows
- Clang 14+, GCC 9+, or Visual Studio 2022+
- Python 3.7+ - For Conan and scripts
- CMake 3.24+ - Build system generator
- Make - Build system (Linux only)
- Conan - Third party C++ library management
- gcovr - Code coverage (Linux only)
- Doxygen - Documentation
- clang-format - Code formatting
- clang-tidy - Linting and static code analysis (Linux only)

### Linux

- Install dependencies (for Ubuntu 22.04 - other Linux distributions should be similar)
  ```sh
  sudo apt install -y gcc-9 g++-9 clang-14 python3 python3-pip cmake make git doxygen clang-format-14 clang-tidy-14 clangd-14 gcovr
  ```
- Install Conan with pip because Conan is not in Ubuntu's package manager
  ```sh
  sudo pip3 install conan
  ```
- Install `cmake-format`
  ```sh
  sudo pip3 install cmake-format
  ```
- Install `autopep8`
  ```sh
  sudo pip3 install autopep8
  ```
- Add symlinks the clang-14 tools so that the correct version is chosen when running `clang-format`, `clang-tidy`, etc
  ```sh
  sudo ln -s /usr/bin/clang-format-14 /usr/bin/clang-format
  sudo ln -s /usr/bin/clang-tidy-14 /usr/bin/clang-tidy
  sudo ln -s /usr/bin/run-clang-tidy-14 /usr/bin/run-clang-tidy
  sudo ln -s /usr/bin/llvm-cov-14 /usr/bin/llvm-cov
  sudo ln -s /usr/bin/clangd-14 /usr/bin/clangd
  ```
- Then refresh the shell so that newly added dependencies are available in the path.
  ```sh
  exec bash
  ```

### Windows

There are two ways to install prerequisites for Windows, [manually](#install-manually) or [with Chocolatey](#install-with-chocolatey). Chocolately is quicker to set up but may conflict with existing installations. We use Chocolatey for CI.

<!-- omit in toc -->
#### Install manually

- Install Visual Studio 2022 Professional: https://visualstudio.microsoft.com/downloads/
  - Select `Desktop Development with C++` and use the default components
- Install Git: https://git-scm.com/downloads
  - Use defaults
- Install LLVM 14.0.6: https://llvm.org/builds
  - When prompted, select `Add LLVM to the system PATH for all users`
- Install CMake: https://cmake.org/download
  - When prompted, select `Add CMake to the system PATH for all users`
- Install Python (version 3.x): https://www.python.org/downloads
  - Select `Add Python 3.x to PATH`
  - Create a symbolic link called `python3.exe` that points to the actual `python` (version 3.x) executable. This is necessary for some of the scripts to run correctly when `#!/usr/bin/env python3` is at the top of the file. Open Command Prompt as administrator and enter:
    ```sh
    where python
    cd <first_path_in_list>
    mklink python3.exe python.exe
    ```
- Install `cmake-format`
  ```sh
  pip3 install cmake-format
  ```
- Install `autopep8`
  ```sh
  pip3 install autopep8
  ```
- Install `colorama` to enable color diff support
  ```sh
  pip3 install colorama
  ```
- Install Conan: https://conan.io/downloads.html
  - Use defaults
- Install Doxygen: https://www.doxygen.nl/download.html
  - After installation, add the install location to your `PATH`. Open PowerShell as administrator and enter:
    ```sh
    [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\doxygen\bin", "Machine")
    ```
- Enable Long Paths. This ensures that all Conan libraries are installed in `~/.conan`. Open PowerShell as administrator and enter:
  ```sh
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```
- Enable MSVC build parallelization. Open PowerShell as administartor and enter:
  ```sh
  [Environment]::SetEnvironmentVariable("UseMultiToolTask", "true", "Machine")
  ```
- Then refresh PowerShell so that newly added dependencies are available in the path.
  ```sh
  refreshenv
  ```

<!-- omit in toc -->
#### Install with Chocolatey

- Install [Chocolatey](https://docs.chocolatey.org/en-us/choco/setup) and then install dependencies
  ```sh
  choco install -y visualstudio2022professional visualstudio2022-workload-nativedesktop python cmake ninja git conan doxygen.install vswhere --installargs 'ADD_CMAKE_TO_PATH=System'
  ```
  ```sh
  choco install -y llvm --version=14.0.6
  ```
  > **Note:** If you see a warning like `Chocolatey detected you are not running from an elevated command shell`, reopen Command Prompt as administrator
- Create a symbolic link called `python3.exe` that points to the actual `python` (version 3.x) executable. This is necessary for some of the scripts to run correctly when `#!/usr/bin/env python3` is at the top of the file.
  ```sh
  where python
  cd <first_path_in_list>
  mklink python3.exe python.exe
  ```
- Install `cmake-format`
  ```sh
  pip3 install cmake-format
  ```
- Install `autopep8`
  ```sh
  pip3 install autopep8
  ```
- Install `colorama` to enable color diff support
  ```sh
  pip3 install colorama
  ```
- Enable Long Paths. This ensures that all Conan libraries are installed correctly. Open PowerShell as administrator and enter:
  ```sh
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```
- Enable MSVC build parallelization. Open PowerShell as administartor and enter:
  ```sh
  [Environment]::SetEnvironmentVariable("UseMultiToolTask", "true", "Machine")
  ```
- Then refresh PowerShell so that newly added dependencies are available in the path.
  ```sh
  refreshenv
  ```

## Clone the repository

```sh
git clone git@github.com:CesiumGS/cesium-omniverse.git --recurse-submodules
```

If you forget the `--recurse-submodules`, nothing will work because the Git submodules will be missing. You should be able to fix it with

```sh
git submodule update --init --recursive
```

## Build

### Linux

```sh
## Release
cmake -B build
cmake --build build --parallel 8

## Debug
cmake -B build-debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel 8
```

Binaries will be written to `build/bin`. Shared libraries and static libraries will be written to `build/lib`.

### Windows

```sh
cmake -B build
cmake --build build --config Release
cmake --build build --config Debug
```

Binaries and shared libraries will be written to `build/bin/Release`. Static libraries and python modules will be written to `build/lib/Release`.

CMake will select the most recent version of Visual Studio on your system unless overridden with a generator (e.g. `-G "Visual Studio 17 2022"`).

### Advanced build options

For faster builds, use the `--parallel` option

```sh
cmake -B build
cmake --build build --parallel 8
```

> **Note:** This option does not work for MSVC builds (see https://gitlab.kitware.com/cmake/cmake/-/issues/20564). Do not use the `--parallel` option since it will override the `UseMultiToolTask` environment variable and revert back to single-threaded compilation.

To use a specific C/C++ compiler, set `CMAKE_CXX_COMPILER` and `CMAKE_C_COMPILER`

```sh
cmake -B build -D CMAKE_CXX_COMPILER=clang++-14 -D CMAKE_C_COMPILER=clang-14
cmake --build build
```

Make sure to use a different build folder for each compiler, otherwise you may see an error from Conan like

```
Library [name] not found in package, might be system one.
```

This error can also be avoided by deleting `build/CMakeCache.txt` before switching compilers.

To view verbose output from the compiler, use the `--verbose` option

```sh
cmake -B build
cmake --build build --verbose
```

To change the build configuration, set `CMAKE_BUILD_TYPE` to one of the following values:

- `Debug`: Required for coverage
- `Release`: Used for production builds
- `RelWithDebInfo`: Similar to `Release` but has debug symbols
- `MinSizeRel`: Similar to `Release` but smaller compile size

On Linux

```sh
cmake -B build-relwithdebinfo -D CMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-relwithdebinfo
```

On Windows

```sh
cmake -B build
cmake --build build --config RelWithDebInfo
```

Note that Windows (MSVC) is a multi-configuration generator meaning all four build configurations are created during the configure step and the specific configuration is chosen during the build step. If using Visual Studio there will be a dropdown to select the build configuration.

Ninja is also supported as an alternative to the MSVC generator. To build with Ninja locally open `x64 Native Tools Command Prompt for VS 2022` and run:

```
cmake -B build -D CMAKE_C_COMPILER=cl -D CMAKE_CXX_COMPILER=cl -G "Ninja Multi-Config"
cmake --build build --config Release --parallel 8
```

### CMake Options

The following CMake options are available for configuring the project.

| Option                            | Description                                                                                                 | Default |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------- |
| `CESIUM_OMNI_ENABLE_TESTS`             | Unit tests                                                                                                  | `ON`    |
| `CESIUM_OMNI_ENABLE_COVERAGE`          | Code coverage for checking the thoroughness of unit tests - only available if `CESIUM_OMNI_ENABLE_TESTS` is `ON` | `ON`    |
| `CESIUM_OMNI_ENABLE_DOCUMENTATION`     | Generate HTML documentation with Doxygen                                                                    | `ON`    |
| `CESIUM_OMNI_ENABLE_SANITIZERS`        | Check for undefined behavior at runtime                                                                     | `OFF`   |
| `CESIUM_OMNI_ENABLE_LINTERS`           | Enable `clang-format` for code formatting and `clang-tidy` for static code analysis                         | `ON`    |
| `CESIUM_OMNI_ENABLE_LINTERS_ON_BUILD`  | Check linting during the build - only available if `CESIUM_OMNI_ENABLE_LINTERS` is `ON`                          | `OFF`   |
| `CESIUM_OMNI_ENABLE_LINTERS_ON_COMMIT` | Check formatting and linting before committing code - only available if `CESIUM_OMNI_ENABLE_LINTERS` is `ON`     | `OFF`   |

For example, to disable unit tests, run

```sh
cmake -B build -D CESIUM_OMNI_ENABLE_TESTS=OFF
cmake --build build
```

## Unit Tests

```sh
cmake -B build
cmake --build build --target tests
ctest --test-dir build
```

Or run the doctest executable directly

```sh
cmake -B build
cmake --build build --target tests
./build/bin/tests
```

## Coverage

It's a good idea to generate code coverage frequently to ensure that you're adequately testing new features. To do so run

```sh
cmake -B build-debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build-debug --target generate-coverage
```

Once finished, the coverage report will be located at `build-debug/coverage/index.html`.

Notes:

- Coverage is disabled in `Release` mode because it would be inaccurate and we don't want coverage instrumentation in our release binaries anyway
- Coverage is not supported on Windows

## Documentation

```sh
cmake -B build
cmake --build build --target generate-documentation
```

Once finished, documentation will be located at `build/docs/html/index.html`.

## Installing

To install `CesiumOmniverse` into the Omniverse Kit extension run:

```sh
cmake -B build
cmake --build build --target install
```

This will install the libraries to `exts/cesium.omniverse/bin`.

To install `CesiumOmniverse` onto the local system run:

```sh
cmake -B build
cmake --build build --target install --component library
```

By default executables will be installed to `/usr/local/bin`, libraries will be installed to `/usr/local/lib`, and includes will be installed to `/usr/local/include`.

To choose a custom install directory set `CMAKE_INSTALL_PREFIX`:

```sh
cmake -B build -D CMAKE_INSTALL_PREFIX=$HOME/Desktop/CesiumOmniverse
cmake --build build --target install --component library
```

## Sanitizers

When sanitizers are enabled they will check for mistakes that are difficult to catch at compile time, such as reading past the end of an array or dereferencing a null pointer. Sanitizers should not be used for production builds because they inject these checks into the binaries themselves, creating some runtime overhead.

Sanitizers

- ASAN - [Address sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)
- UBSAN - [Undefined behavior sanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)

> **Note:** memory leak detection is not supported on Windows currently. See https://github.com/google/sanitizers/issues/1026#issuecomment-850404983

> **Note:** memory leak detection does not work while debugging with gdb. See https://stackoverflow.com/questions/54022889/leaksanitizer-not-working-under-gdb-in-ubuntu-18-04

To verify that sanitization is working, add the following code to any cpp file.

```c++
int arr[4] = {0};
arr[argc + 1000] = 0;
```

After running, it should print something like

```
main.cpp:114:22: runtime error: index 1001 out of bounds for type 'int [4]'
main.cpp:114:24: runtime error: store to address 0x7ffe16f44c44 with insufficient space for an object of type 'int'
0x7ffe16f44c44: note: pointer points here
  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00
              ^
```

## Formatting

To format code based on the [`.clang-format`](./.clang-format) configuration file

```sh
cmake -B build
cmake --build build --target clang-format-fix-all
```

The full list of targets is below:

- `clang-format-fix-all` - Formats all code
- `clang-format-fix-staged` - Formats staged code
- `clang-format-check-all` - Checks for formatting problems in all code
- `clang-format-check-staged` - Checks for formatting problems in staged code

Please note that the `clang-format-fix-all` and `clang-format-fix-staged` targets will add fixes in the working area, not in the staging area. We also have a Git hook that is installed on project configuration that will check if the staging area is properly formatted before permitting a commit.

## Linting

`clang-tidy` is run during the build to catch common coding errors. `clang-tidy` is used for linting and static code analysis based on the [`.clang-tidy`](./.clang-tidy) configuration file.

We also generate CMake targets to run these tools manually

Run `clang-tidy`:

```sh
cmake -B build
cmake --build build --target clang-tidy
```

Note that these targets are not available on Windows currently.

## VSCode

We use VSCode as our primary IDE for development. While everything can be done on the command line the `.vscode` project folder has built-in tasks for building, running unit tests, generating documentation, etc.

### Workspaces

Each workspace contains recommended extensions and settings for VSCode development. Make sure to open the workspace for your OS instead of opening the `cesium-omniverse` folder directly.

- [cesium-omniverse-linux.code-workspace](./.vscode/cesium-omniverse-linux.code-workspace)
- [cesium-omniverse-windows.code-workspace](./.vscode/cesium-omniverse-windows.code-workspace)

### Debugging

- Create `.vscode/launch.json` and copy the configurations below. Feel free to edit these or add additional configurations for your own debugging purposes.
- Select an option from the `Run and Debug` panel, such as `Test`, and click the green arrow.

<!-- omit in toc -->
#### Linux

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Kit App",
      "preLaunchTask": "Build Only (debug)",
      "program": "${workspaceFolder}/extern/nvidia/app/kit/kit",
      "args": [
        "${workspaceFolder}/apps/cesium.omniverse.app.kit"
      ],
      "cwd": "${workspaceFolder}",
      "type": "lldb",
      "request": "launch",
      "console": "internalConsole",
      "internalConsoleOptions": "openOnSessionStart",
      "MIMode": "gdb",
      "setupCommands": [
        {
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        },
        {
          "text": "set print elements 0"
        }
      ]
    },
    {
      "name": "Code",
      "preLaunchTask": "Build Only (debug)",
      "program": "${workspaceFolder}/extern/nvidia/app/kit/kit",
      "args": [
        "${workspaceFolder}/extern/nvidia/app/apps/omni.code.kit"
      ],
      "cwd": "${workspaceFolder}",
      "type": "lldb",
      "request": "launch",
      "console": "internalConsole",
      "internalConsoleOptions": "openOnSessionStart",
      "MIMode": "gdb",
      "setupCommands": [
        {
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        },
        {
          "text": "set print elements 0"
        }
      ]
    },
    {
      "name": "Test",
      "preLaunchTask": "Build Only (debug)",
      "program": "${workspaceFolder}/build-debug/bin/tests",
      "env": {
        // Disable LSAN when debugging since it doesn't work with GDB and prints harmless but annoying warning messages
        "ASAN_OPTIONS": "detect_leaks=0",
        "UBSAN_OPTIONS": "print_stacktrace=1"
      },
      "cwd": "${workspaceFolder}",
      "type": "lldb",
      "request": "launch",
      "console": "internalConsole",
      "internalConsoleOptions": "openOnSessionStart",
      "MIMode": "gdb",
      "setupCommands": [
        {
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        },
        {
          "text": "set print elements 0"
        }
      ]
    }
  ]
}
```

<!-- omit in toc -->
#### Windows

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Kit App",
      "preLaunchTask": "Build Only (debug)",
      "program": "${workspaceFolder}/extern/nvidia/app/kit/kit.exe",
      "args": [
        "${workspaceFolder}/apps/cesium.omniverse.app.kit"
      ],
      "cwd": "${workspaceFolder}",
      "type": "cppvsdbg",
      "request": "launch",
      "console": "internalConsole",
      "internalConsoleOptions": "openOnSessionStart"
    },
    {
      "name": "Code",
      "preLaunchTask": "Build Only (debug)",
      "program": "${workspaceFolder}/extern/nvidia/app/kit/kit.exe",
      "args": [
        "${workspaceFolder}/extern/nvidia/app/apps/omni.code.kit"
      ],
      "cwd": "${workspaceFolder}",
      "type": "cppvsdbg",
      "request": "launch",
      "console": "internalConsole",
      "internalConsoleOptions": "openOnSessionStart"
    },
    {
      "name": "Test",
      "preLaunchTask": "Build Only (debug)",
      "program": "${workspaceFolder}/build/bin/Debug/tests",
      "cwd": "${workspaceFolder}",
      "type": "cppvsdbg",
      "request": "launch",
      "console": "internalConsole",
      "internalConsoleOptions": "openOnSessionStart"
    }
  ]
}
```

### Tasks

[`.vscode/tasks.json`](./.vscode/tasks.json) comes with the following tasks:

- Configure - configures the project
- Build (advanced) - configures and builds the project
- Build (verbose) - configures and builds the project with verbose output
- Build (debug) - configures and builds the project in debug mode with the default compiler
- Build (release) - configures and builds the project in release mode with the default compiler
- Build Only (debug) - builds the project in debug mode with the default compiler
- Build Only (release) - builds the project in release mode with the default compiler
- Clean - cleans the build directory
- Test - runs unit tests
- Coverage - generates a coverage report and opens a web browser showing the results
- Documentation - generates documentation and opens a web browser showing the results
- Format - formats the code with clang-format
- Lint - runs clang-tidy
- Lint Fix - runs clang-tidy and fixes issues
- Dependency graph - shows the third party library dependency graph

To run a task:

- `Ctrl + Shift + B` and select the task, e.g. `Build`
- Select the build type and compiler (if applicable)

## Project Structure

- `src` - Source code for the CesiumOmniverse library
- `include` - Include directory for the CesiumOmniverse library
- `tests` - Unit tests
- `extern` - Third-party dependencies that aren't on Conan
- `cmake` - CMake helper files
- `scripts` - Build scripts and Git hooks

## Third Party Libraries

We use [Conan](https://conan.io/) as our C++ third party package manager for dependencies that are public and not changed often. Third party libraries are always built from source and are cached on the local system for subsequent builds.

To add a new dependency to Conan

- Add it to [AddConanDependencies.cmake](./cmake/AddConanDependencies.cmake)
- Call `find_package` in [CMakeLists.txt](./CMakeLists.txt)
- Add the library to the `LIBRARIES` field in any relevant `setup_lib` or `setup_app` calls

Some dependencies are pulled in as Git submodules instead. When adding a new git submodule add the license to [ThirdParty.extra.json](./ThirdParty.extra.json).

[ThirdParty.json](./ThirdParty.json) is autogenerated and combines [ThirdParty.extra.json](./ThirdParty.extra.json) and Conan dependencies.
