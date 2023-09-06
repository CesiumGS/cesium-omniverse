#include "cesium/omniverse/CudaManager.h"

#include "cesium/omniverse/CudaKernels.h"

#include <omni/gpucompute/GpuCompute.h>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace cesium::omniverse {

void CudaManager::initialize() {
    if (_initialized) {
        return;
    }

    CUresult result;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("ERROR: CUDA did not init");
    }

    result = cuDeviceGet(&_device, 0);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("ERROR: CUDA did not get a device");
    }

    result = cuCtxCreate(&_context, 0, _device);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("ERROR: could not create CUDA context");
    }

    _initialized = true;
}

void CudaManager::runAllRunners() {
    // for (auto& [updateType, runners] : _runnersByUpdateType) {
    //     if (updateType == CudaUpdateType::ONCE) {
    //         throw std::runtime_error("Single-run kernels are not yet supported.");
    //     } else if (updateType == CudaUpdateType::ON_UPDATE_FRAME) {
    //         for (auto& [tileId, runner] : runners) {
    //             runRunner(*runner);
    //         }
    //     }
    // }
}

#pragma warning(push)
#pragma warning(disable: 4100)
void CudaManager::createRunner(CudaKernelType cudaKernelType, CudaUpdateType cudaUpdateType, int64_t tileId, CudaKernelArgs kernelArgs, int numberOfElements) {
    if (_kernels.find(cudaKernelType) == _kernels.end()) {
        compileKernel(cudaKernelType);
    }

    auto runnerPtr = std::make_unique<CudaRunner>(cudaKernelType, cudaUpdateType, tileId, kernelArgs, static_cast<int>(numberOfElements));
    auto& innerMap = _runnersByUpdateType[cudaUpdateType];
    innerMap.insert({tileId, std::move(runnerPtr)});
}
#pragma warning(pop)


void CudaManager::onUpdateFrame() {
    runAllRunners();
}



void CudaManager::removeRunner(int64_t tileId) {
    for (auto& [updateType, runners] : _runnersByUpdateType) {
        runners.erase(tileId);
    }
}


void CudaManager::runRunner(CudaRunner& runner) {
    for (size_t bucketNumber = 0; bucketNumber != runner.primBucketList.bucketCount(); bucketNumber++)
    {
        auto kernel = _kernels[runner.kernelType];

        int minGridSize, blockSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel.function, nullptr, 0, 0);
        int numBlocks = (static_cast<int>(runner.elementCount) + blockSize - 1) / blockSize;

        // void* tempArgs[] = {&runner.elementCount}; //NOLINT

        // (quad** quads, double3* lookAtPosition, float3* lookAtUp, float *quadSize, int numQuads)
        glm::dvec3 lookAtPosition(0, 0, 0); // placeholder
        glm::fvec3 lookAtUp(0, 1.0f, 0);
        float quadSize = 1.0f;
        void *args[] = { &runner.quadBucketMap[bucketNumber], &lookAtPosition, &lookAtUp, &quadSize, &runner.elementCount}; //NOLINT
        //auto args = runner.getPackedKernelArgs(bucketNumber);
        // runner.setPackedKernelArgs(bucketumber, args);
        auto launchResult =
            cuLaunchKernel(kernel.function, numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (launchResult) {
            const char* errName = nullptr;
            const char* errString = nullptr;

            cuGetErrorName(launchResult, &errName);
            cuGetErrorString(launchResult, &errString);

            std::cout << "Error launching kernel: " << errName << ": " << errString << std::endl;

            CUcontext currentContext;
            cuCtxGetCurrent(&currentContext);
            // There's currently an issue where the CUDA context occassionally
            // changes. We have to test for and handle this.
            if (currentContext != _context) {
                cuCtxSetCurrent(_context);
            }
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

    nvrtcResult compileResult = nvrtcCompileProgram(kernel.program, 0, nullptr);
    std::unique_ptr<char[]> log; //NOLINT
    if (compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(kernel.program, &logSize);
        log.reset(new char[logSize]);
        nvrtcGetProgramLog(kernel.program, log.get());
        throw std::runtime_error(log.get());
    }

    // Get the PTX (assembly code for the GPU)
    size_t ptxSize;
    nvrtcGetPTXSize(kernel.program, &ptxSize);
    kernel.ptx = new char[ptxSize];
    nvrtcGetPTX(kernel.program, kernel.ptx);

    cuModuleLoadDataEx(&kernel.module, kernel.ptx, 0, nullptr, nullptr);
    auto cudaResult = cuModuleGetFunction(&kernel.function, kernel.module, kernelFunctionName);
    if (cudaResult != CUDA_SUCCESS) {
        const char* errName = nullptr;
        const char* errString = nullptr;
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
        case CudaKernelType::LOOKAT_QUADS:
            return cesium::omniverse::cudaKernels::lookAtQuadsKernel;
            break;
        case CudaKernelType::CREATE_VOXELS:
            return cesium::omniverse::cudaKernels::createVoxelsKernel;
            break;
        case CudaKernelType::HELLO_WORLD:
            return cesium::omniverse::cudaKernels::helloWorldKernel;
        default:
            throw new std::runtime_error("Attempt to compile an unsupported CUDA kernel.");
    }
}

[[nodiscard]] const char* CudaManager::getFunctionName(CudaKernelType kernelType) const {
    switch (kernelType) {
        case CudaKernelType::LOOKAT_QUADS:
            return "lookAtQuads";
            break;
        case CudaKernelType::CREATE_VOXELS:
            return "createVoxels";
            break;
        case CudaKernelType::HELLO_WORLD:
            return "helloWorld";
            break;
        default:
            throw new std::runtime_error("Attempt to find function for an unsupported CUDA kernel.");
    }
}

omni::fabric::Token CudaManager::getTileToken(int64_t tileId) {
    if (_tileTokens.find(tileId) == _tileTokens.end()) {
        auto tokenName = "tile" + std::to_string(tileId);
        _tileTokens[tileId] = omni::fabric::Token(tokenName.c_str());
    }

    return _tileTokens[tileId];
}
} // namespace cesium::omniverse
