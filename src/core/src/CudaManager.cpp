#include "cesium/omniverse/CudaManager.h"
#include "cesium/omniverse/CudaKernels.h"

#include <omni/gpucompute/GpuCompute.h>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

namespace cesium::omniverse {

    void CudaManager::initialize() {
        if (_initialized) return;

        CUresult result;

        result = cuInit(0);
        if (result != CUDA_SUCCESS) {
            // std::cout << "error: CUDA did not init." << std::endl; //TODO: logger
            throw std::runtime_error("ERROR");
        }

        result = cuDeviceGet(&_device, 0);
        if (result != CUDA_SUCCESS) {
            // std::cout << "error: CUDA did not get a device." << std::endl; //TODO: logger
            throw std::runtime_error("ERROR");
        }

        result = cuCtxCreate(&_context, 0, _device);
        if (result != CUDA_SUCCESS) {
            // std::cout << "error: could not create CUDA context." << std::endl; //TODO: logger
            throw std::runtime_error("ERROR");
        }

        _initialized = true;
    }

    void CudaManager::runAllRunners() {
        for (auto& [updateType, runners] : _runnersByUpdateType) {
            if (updateType == CudaUpdateType::ONCE) {
                throw std::runtime_error("Single-run kernels are not yet supported.");
            }
            else if (updateType == CudaUpdateType::ON_UPDATE_FRAME) {
                auto onceRunners = runners;
                for (auto& [tileId, runner] : runners) {
                    runRunner(runner);
                }
            }
        }
    }

    void CudaManager::onUpdateFrame() {
        runAllRunners();
    }

    void CudaManager::addRunner(CudaRunner& cudaRunner) {
        if (!_initialized) initialize();

        if (_kernels.find(cudaRunner.kernelType) == _kernels.end()) {
            compileKernel(cudaRunner.kernelType);
        }

        auto& innerMap = _runnersByUpdateType[cudaRunner.getUpdateType()];
        // innerMap[cudaRunner.getTileId()] = cudaRunner;
        innerMap.insert({cudaRunner.getTileId(), std::move(cudaRunner)});

    }

    void CudaManager::removeRunner(int64_t tileId) {
        for (auto& [updateType, runners] : _runnersByUpdateType) {
            runners.erase(tileId);
        }
    }

void** CudaManager::packArgs(CudaKernelArgs cudaKernelArgs, CudaKernelType cudaKernelType) {
    void **args = new void*[10](); // dynamically allocate an array of 10 void pointers and zero-initialize them

    switch (cudaKernelType) {
        case CudaKernelType::CREATE_VOXELS:
            args[0] = &cudaKernelArgs.args["points"];
            break;
        case CudaKernelType::HELLO_WORLD:
            // nothing to do as we zero-initialized the array
            break;
        default:
            delete[] args; // free the allocated memory before throwing an exception
            throw std::runtime_error("Cannot create kernel args\n");
    }

    return args;
}

    void CudaManager::runRunner(CudaRunner& runner) {
        auto kernel = _kernels[runner.kernelType];

        int minGridSize, blockSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel.function, nullptr, 0, 0);
        int numBlocks = (static_cast<int>(runner.elementCount) + blockSize - 1) / blockSize;

        void *tempArgs[] = { &runner.elementCount}; //NOLINT
        auto launchResult = cuLaunchKernel(kernel.function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, tempArgs, nullptr);
        if (launchResult) {
            const char *errName = nullptr;
            const char *errString = nullptr;

            cuGetErrorName(launchResult, &errName);
            cuGetErrorString(launchResult, &errString);

            std::cout << "Error launching kernel: " << errName << ": " << errString << std::endl;

            CUcontext currentContext;
            cuCtxGetCurrent(&currentContext);
            if (currentContext != _context) {
                // std::cout << "Warning: Context has changed!" << std::endl;
                // auto threadId = std::this_thread::get_id();
                // std::cout << "Current thread ID: " << threadId << std::endl;

                // throw std::runtime_error("contexts don't match");
                cuCtxSetCurrent(_context);
            }
        }
    }

    void CudaManager::compileKernel(CudaKernelType kernelType) {
        if (_kernels.find(kernelType) != _kernels.end()) {
            return;
        }

        auto kernelCode = getKernelCode(kernelType);
        auto kernelFunctionName = getFunctionName(kernelType);

        CudaKernel kernel;

        nvrtcCreateProgram(&kernel.program, kernelCode, kernelFunctionName, 0, nullptr, nullptr);

        nvrtcResult res = nvrtcCompileProgram(kernel.program, 0, nullptr);
        if (res != NVRTC_SUCCESS) {
            size_t logSize;
            nvrtcGetProgramLogSize(kernel.program, &logSize);
            char* log = new char[logSize];
            nvrtcGetProgramLog(kernel.program, log);
            throw std::runtime_error(log);
        }

        // Get the PTX (assembly code for the GPU)
        size_t ptxSize;
        nvrtcGetPTXSize(kernel.program, &ptxSize);
        kernel.ptx = new char[ptxSize];
        nvrtcGetPTX(kernel.program, kernel.ptx);

        cuModuleLoadDataEx(&kernel.module, kernel.ptx, 0, nullptr, nullptr);
        auto cudaResult = cuModuleGetFunction(&kernel.function, kernel.module, kernelFunctionName);
        if (cudaResult != CUDA_SUCCESS) {
            const char *errName = nullptr;
            const char *errString = nullptr;
            cuGetErrorName(cudaResult, &errName);
            cuGetErrorString(cudaResult, &errString);

            std::ostringstream errMsg;
            errMsg << "Error getting function: " << errName << ": " << errString;
            throw std::runtime_error(errMsg.str());
        }

        _kernels[kernelType] = kernel;
    }

    [[nodiscard]] const char* CudaManager::getKernelCode(CudaKernelType kernelType) const {
        switch (kernelType) {
            case CudaKernelType::CREATE_VOXELS:
                return cesium::omniverse::cudaKernels::createVoxelsKernel;
                break;
            case CudaKernelType::HELLO_WORLD:
                return cesium::omniverse::cudaKernels::helloWorldKernel;
            default:
                throw new std::exception("Attempt to compile an unsupported CUDA kernel.");
        }
    }

    [[nodiscard]] const char* CudaManager::getFunctionName(CudaKernelType kernelType) const {
        switch (kernelType) {
            case CudaKernelType::CREATE_VOXELS:
                return "createVoxels";
                break;
            case CudaKernelType::HELLO_WORLD:
                return "helloWorld";
                break;
            default:
                throw new std::exception("Attempt to find function for an unsupported CUDA kernel.");
        }
    }
}
