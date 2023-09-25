#include "cesium/omniverse/CudaManager.h"

#include "cesium/omniverse/CudaKernels.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/gpucompute/GpuCompute.h>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace cesium::omniverse {

#pragma warning( push )
#pragma warning( disable : 4100 )  // Disable warning C4100: 'identifier' : unreferenced


bool hasRun = false;
// glm::dvec3 lookAtPosition(0, 0, 0); // placeholder
glm::dvec3 lookatPositionHost{0.0, 0.0, 0.0};
glm::fvec3 lookatUpHost{0.0, 1.0, 0.0};
// glm::fvec3 lookAtUp(0, 1.0f, 0);
float quadSizeHost = 0.75f;
// float quadSize = 1.0f;
float testFloat = 123.45f;


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
    if (hasRun) {
        return;
    }

    for (auto& [updateType, runners] : _runnersByUpdateType) {
        if (updateType == CudaUpdateType::ONCE) {
            throw std::runtime_error("Single-run kernels are not yet supported.");
        } else if (updateType == CudaUpdateType::ON_UPDATE_FRAME) {
            for (auto& [tileId, runner] : runners) {
                runRunner(*runner);
                hasRun = true;
            }
            hasRun = true;
        }
    }

}

void CudaManager::createRunner(CudaKernelType cudaKernelType, CudaUpdateType cudaUpdateType, int64_t tileId, CudaKernelArgs kernelArgs, int numberOfElements) {
    if (_kernels.find(cudaKernelType) == _kernels.end()) {
        compileKernel(cudaKernelType);
    }

    auto runnerPtr = std::make_unique<CudaRunner>(cudaKernelType, cudaUpdateType, tileId, kernelArgs, static_cast<int>(numberOfElements));
    auto& innerMap = _runnersByUpdateType[cudaUpdateType];
    innerMap.insert({tileId, std::move(runnerPtr)});
}


void CudaManager::onUpdateFrame() {
    runAllRunners();
}


void CudaManager::removeRunner(int64_t tileId) {
    for (auto& [updateType, runners] : _runnersByUpdateType) {
        runners.erase(tileId);
    }
}

#pragma warning( push )
#pragma warning( disable : 4100 )
void CudaManager::runRunner(CudaRunner& runner) {

    // BEGIN SPLICED CODE
    // auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    // auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    // auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    // auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    // // tile ID for DEBUG purposes
    // omni::fabric::AttrNameAndType primTag(CudaManager::getInstance().getTileTokenType(), CudaManager::getInstance().getTileToken(0));
    // auto bucketList = stageReaderWriter.findPrims({primTag});

    // std::cout << "numBuckets " << bucketList.bucketCount() << std::endl;
    // for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    // {
    //     auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f>(bucketList, bucketNum, FabricTokens::points);
    //     if (positions.data() != nullptr) {
    //         std::cout << "no issue running PoC code" << std::endl;
    //     }
    // }


    // auto srw = UsdUtil::getFabricStageReaderWriter();
    // auto tileToken = CudaManager::getInstance().getTileToken(runner.getTileId());
    // omni::fabric::AttrNameAndType primTag(CudaManager::getInstance().getTileTokenType(), tileToken);
    // omni::fabric::PrimBucketList bucketList = srw.findPrims({primTag});
    // for (size_t bucketNumber = 0; bucketNumber != bucketList.bucketCount(); bucketNumber++)
    // {
    //     auto positions = srw.getArrayAttributeArray<pxr::GfVec3f>(bucketList, bucketNumber, FabricTokens::points);
    //     if (positions.data() != nullptr) {
    //         std::cout << "no issue running PoC code" << std::endl;
    //     }
    // }


    // END SPLICED CODE



    // BEGIN POC

    // auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    // auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    // auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    // auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    // // tile ID for DEBUG purposes
    // omni::fabric::AttrNameAndType primTag(CudaManager::getInstance().getTileTokenType(), CudaManager::getInstance().getTileToken(0));
    // auto bucketList = stageReaderWriter.findPrims({primTag});

    // std::cout << "numBuckets " << bucketList.bucketCount() << std::endl;
    // for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    // {
    //     auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f>(bucketList, bucketNum, FabricTokens::points);
    //     if (positions.data() != nullptr) {
    //         std::cout << "no issue running PoC code" << std::endl;
    //     }
    // }



    // END POC

    // BEGIN MEM-ISSUE CODE

    auto srw = UsdUtil::getFabricStageReaderWriter();

    auto tileToken = CudaManager::getInstance().getTileToken(runner.getTileId());


    omni::fabric::AttrNameAndType primTag(CudaManager::getInstance().getTileTokenType(), tileToken);
    omni::fabric::PrimBucketList bucketList = srw.findPrims({primTag});

    auto lookatPositionDevice = allocAndCopyToDevice(&lookatPositionHost, sizeof(glm::dvec3));
    auto lookatUpDevice = allocAndCopyToDevice(&lookatUpHost, sizeof(glm::fvec3));
    auto quadSizeDevice = allocAndCopyToDevice(&quadSizeHost, sizeof(float));


    // CUdeviceptr quadSizeDevice;
    // auto err = cuMemAlloc(&quadSizeDevice, sizeof(float));
    // if (err != CUDA_SUCCESS) {
    //     const char *errName;
    //     const char *errStr;
    //     cuGetErrorName(err, &errName);
    //     cuGetErrorString(err, &errStr);
    //     printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
    //     return;
    // }

    // err = cuMemcpyHtoD(quadSizeDevice, &quadSizeHost, sizeof(float));
    // if (err != CUDA_SUCCESS) {
    //     const char *errName;
    //     const char *errStr;
    //     cuGetErrorName(err, &errName);
    //     cuGetErrorString(err, &errStr);
    //     printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
    //     return;
    // }

    for (size_t bucketNumber = 0; bucketNumber != bucketList.bucketCount(); bucketNumber++)
    {
        auto kernel = _kernels[runner.kernelType];

        int minGridSize, blockSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel.function, nullptr, 0, 0);
        int numBlocks = (static_cast<int>(runner.elementCount) + blockSize - 1) / blockSize;

        // void* tempArgs[] = {&runner.elementCount}; //NOLINT

        // (quad** quads, double3* lookAtPosition, float3* lookAtUp, float *quadSize, int numQuads)
        // glm::dvec3 lookAtPosition(0, 0, 0); // placeholder
        // glm::fvec3 lookAtUp(0, 1.0f, 0);
        // float quadSize = 1.0f;

        // void *args[] = { &runner.quadBucketMap[bucketNumber], &lookAtPosition, &lookAtUp, &quadSize, &runner.elementCount}; // NOLINT

        // TODO: getArrayAttributeArrayWr
        gsl::span<pxr::GfVec3f> positions = srw.getAttributeArrayGpu<pxr::GfVec3f>(bucketList, bucketNumber, FabricTokens::points);
        pxr::GfVec3f* d = positions.data();
        // void *args[] = { &positions, &lookatPositionHost, &lookatUpHost, &quadSizeHost, &runner.elementCount}; // NOLINT
        // auto args = runner.getPackedKernelArgs(bucketNumber);
        // runner.setPackedKernelArgs(bucketumber, args);
        // void *args[] = { &_quadSizeHost, &runner.elementCount }; // NOLINT

        // void *args[] = { &d, &runner.elementCount}; // NOLINT
        void *args[] = { &d, &lookatPositionDevice, &lookatUpDevice, &quadSizeDevice, &runner.elementCount}; // NOLINT


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
        } else {
            std::cout << "got args and launched kernel fine" << std::endl;
        }
    }

    // err = cuMemFree(quadSizeDevice);
    // if (err != CUDA_SUCCESS) {
    //     const char *errName;
    //     const char *errStr;
    //     cuGetErrorName(err, &errName);
    //     cuGetErrorString(err, &errStr);
    //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
    //     return;
    // }

    freeDeviceMemory(lookatPositionDevice);
    freeDeviceMemory(lookatUpDevice);
    freeDeviceMemory(quadSizeDevice);
}
#pragma warning( pop )

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
        case CudaKernelType::PRINT_FLOAT:
            return cesium::omniverse::cudaKernels::printFloatKernel;
        case CudaKernelType::PRINT_POINTS:
            return cesium::omniverse::cudaKernels::printPointsKernel;
        case CudaKernelType::PRINT_QUADS:
            return cesium::omniverse::cudaKernels::printQuadsKernel;
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
        case CudaKernelType::PRINT_FLOAT:
            return "printFloat";
            break;
        case CudaKernelType::PRINT_POINTS:
            return "printPoints";
            break;
        case CudaKernelType::PRINT_QUADS:
            return "run";
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

CUdeviceptr CudaManager::allocAndCopyToDevice(void* hostPtr, size_t size) {
    CUdeviceptr devicePtr;
    CudaError::check(cuMemAlloc(&devicePtr, size), "cuMemAlloc");
    CudaError::check(cuMemcpyHtoD(devicePtr, hostPtr, size), "cuMemcpyHtoD");
    return devicePtr;
}

void CudaManager::freeDeviceMemory(CUdeviceptr devicePtr) {
    CUresult err = cuMemFree(devicePtr);
    if (err != CUDA_SUCCESS) {
        const char* errName;
        const char* errStr;
        cuGetErrorName(err, &errName);
        cuGetErrorString(err, &errStr);
        printf("cuMemFree failed for address %p: %s: %s\n", (void*)devicePtr, errName, errStr);
    }
}

void CudaManager::runProofOfConceptCode() {
    glm::fvec3 target{0, 0, 0};
    glm::fvec3 targetUp{0, 1.0f, 0};
    billboardMultiQuadCuda(target, targetUp);
}

omni::fabric::Token CudaManager::getCudaTestAttributeFabricToken() {
    static const auto cudaTestAttributeFabricToken = omni::fabric::Token("cudaTest");
    return cudaTestAttributeFabricToken;
}

void CudaManager::billboardMultiQuadCuda(glm::fvec3 lookatPosition2, glm::fvec3 lookatUp2) {

    //get all prims with the custom attr
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
    auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
    auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);

    // tile ID for DEBUG purposes
    omni::fabric::AttrNameAndType primTag(CudaManager::getInstance().getTileTokenType(), CudaManager::getInstance().getTileToken(0));
    auto bucketList = stageReaderWriter.findPrims({primTag});

    std::cout << "numBuckets " << bucketList.bucketCount() << std::endl;
    for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
    {
        auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
        if (positions.data() != nullptr) {
            std::cout << "no issue running PoC code" << std::endl;
        }
    }

    // std::cout << "modified " << primCount << " quads" << std::endl;

    // err = cuMemFree(lookatPositionDevice);
    // if (err != CUDA_SUCCESS) {
    //     const char *errName;
    //     const char *errStr;
    //     cuGetErrorName(err, &errName);
    //     cuGetErrorString(err, &errStr);
    //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
    //     return;
    // }

    // err = cuMemFree(lookatUpDevice);
    // if (err != CUDA_SUCCESS) {
    //     const char *errName;
    //     const char *errStr;
    //     cuGetErrorName(err, &errName);
    //     cuGetErrorString(err, &errStr);
    //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
    //     return;
    // }

    // err = cuMemFree(quadSizeDevice);
    // if (err != CUDA_SUCCESS) {
    //     const char *errName;
    //     const char *errStr;
    //     cuGetErrorName(err, &errName);
    //     cuGetErrorString(err, &errStr);
    //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
    //     return;
    // }
}
#pragma warning( pop )

} // namespace cesium::omniverse
