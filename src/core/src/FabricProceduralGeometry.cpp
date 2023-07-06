#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

// #include <carb/flatcache/FlatCache.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <carb/Framework.h>

// #include <carb/flatcache/FlatCacheUSD.h>
// #include <omni/usd/omni.h>
// #include <omni/usd/UsdContextIncludes.h>
// #include <omni/usd/UsdContext.h>

// #include <CesiumUsdSchemas/data.h>
#include "pxr/base/tf/token.h"

#include <pxr/usd/usd/prim.h>
// #include <omni/gpucompute/GpuCompute.h>
#include <iostream>
#include <omni/gpucompute/GpuCompute.h>



int cesium::omniverse::FabricProceduralGeometry::createCube() {
    //modifyUsdPrim();
    //modify1000Prims();
    modify1000PrimsViaCuda();

    return 178;
}

void cesium::omniverse::FabricProceduralGeometry::modifyUsdPrim() {
    //Linker error getting UsdContext using omni::usd
    // auto context = omni::usd::UsdContext::getContext();
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    //carb::flatcache::StageInProgress stageInProgress = UsdUtil::getFabricStageInProgress();
    omni::fabric::StageReaderWriter stageReaderWriter = UsdUtil::getFabricStageReaderWriter();
    //auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    long id = Context::instance().getStageId();
    //auto usdStageId = carb::flatcache::UsdStageId{static_cast<uint64_t>(id)};
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};

    //create a cube in USD and set its size.
    pxr::UsdPrim prim = usdStagePtr->DefinePrim(pxr::SdfPath("/TestCube"), pxr::TfToken("Cube"));
    prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(3.0);

    //create a second cube in USD as a reference.
    // pxr::UsdPrim prim2 = usdStagePtr->DefinePrim(pxr::SdfPath("/TestCube2"), pxr::TfToken("Cube"));
    // prim2.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(6.0);

    //prefetch it to Fabric’s cache.
    //carb::flatcache::Path primPath("/TestCube");
    omni::fabric::Path primPath("/TestCube");
    iStageReaderWriter->prefetchPrim(usdStageId, primPath);

    //use Fabric to modify the cube’s dimensions
    //auto sizeFabricToken = carb::flatcache::Token("size");
    auto sizeFabricToken = omni::fabric::Token("size");

    //double& size = *stageInProgress.getAttribute<double>(primPath, carb::flatcache::Token("size"));
    double& size = *stageReaderWriter.getAttribute<double>(primPath, omni::fabric::Token("size"));
    double sizeTarget = 30;
    size = sizeTarget;

    // //write our changes back to USD.
    // const auto flatCache = carb::getCachedInterface<carb::flatcache::FlatCache>();
    // if (flatCache->createCache != nullptr) {
    //     auto& pathToAttributesMap = flatCache->createCache(usdStageId, flatcache::kDefaultUserId, carb::flatcache::CacheType::eWithoutHistory);
    //     flatCache->cacheToUsd(pathToAttributesMap);
    // }
    const auto iFabricUsd = carb::getCachedInterface<omni::fabric::IFabricUsd>();
    auto fabricId = omni::fabric::FabricId();
    iFabricUsd->exportUsdPrimData(fabricId);


    //check that Fabric correctly modified the USD stage.
    pxr::UsdAttribute sizeAttr = prim.GetAttribute(pxr::TfToken("size"));
    double value;
    sizeAttr.Get(&value);
    if (value == sizeTarget) {
        std::cout << "modified stage" << std::endl;
    } else {
        std::cout << "did not modify stage" << std::endl;
    }
}

void cesium::omniverse::FabricProceduralGeometry::modify1000Prims() {
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();

    //use USD to make a thousand cubes
    const size_t cubeCount = 1000;
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(1.0);
    }


    //call prefetchPrim to get the data into Fabric.
    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    //tell Fabric which prims to change. Select all prims of type Cube.
    omni::fabric::AttrNameAndType cubeTag(
        omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName),
        omni::fabric::Token("Cube"));

    const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(id)});
    auto fabricReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::PrimBucketList cubeBuckets = fabricReaderWriter.findPrims({ cubeTag });
    //carb::flatcache::PrimBucketList cubeBuckets = iStageInProgress->findPrims({ cubeTag });
    //carb::flatcache::PrimBucketList cubeBuckets = stage.findPrims({ cubeTag });

    // Fabric is free to store the 1000 cubes in as many buckets as it likes...iterate over the buckets
    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        auto sizes = fabricReaderWriter.getAttributeArray<double>(cubeBuckets, bucket, omni::fabric::Token("size"));
        for (double& size : sizes)
        {
            size *= 10;
        }
    }
}

void cesium::omniverse::FabricProceduralGeometry::modify1000PrimsViaCuda() {
    const size_t cubeCount = 1000;

    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    for (size_t i = 0; i != cubeCount; i++)
    {
        pxr::SdfPath path("/cube_" + std::to_string(i));
        pxr::UsdPrim prim = usdStagePtr->DefinePrim(path, pxr::TfToken("Cube"));
        prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(1.0);
    }

    long id = Context::instance().getStageId();
    auto usdStageId = omni::fabric::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        omni::fabric::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageReaderWriter->prefetchPrim(usdStageId, path);
    }

    omni::fabric::AttrNameAndType cubeTag(omni::fabric::Type(omni::fabric::BaseDataType::eTag, 1, 0, omni::fabric::AttributeRole::ePrimTypeName), omni::fabric::Token("Cube"));
    const auto stageReaderWriterId = iStageReaderWriter->get(omni::fabric::UsdStageId{static_cast<uint64_t>(id)});
    auto fabricReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
    omni::fabric::PrimBucketList cubeBuckets = fabricReaderWriter.findPrims({ cubeTag });

    auto isCudaCompatible = checkCudaCompatibility();
    if (!isCudaCompatible) {
        std::cout << "error: CUDA drives and toolkit versions are not compatible." << std::endl;
    }

    //CUDA via CUDA_JIT and string
    static const char* scaleCubes =
        "   extern \"C\" __global__"
        "   void scaleCubes(double* cubeSizes, size_t count)"
        "   {"
        "       size_t i = blockIdx.x * blockDim.x + threadIdx.x;"
        "       if(count<=i) return;"
        ""
        "       cubeSizes[i] *= 10.0;"
        "   }";

    //minimally viable test kernel
    // static const char* scaleCubes =
    // "extern \"C\" __global__ void testKernel(int* output)"
    // "{"
    // "    int i = blockIdx.x * blockDim.x + threadIdx.x;"
    // "    output[i] = i;"
    // "}";

    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not init." << std::endl;
    }

    CUdevice device;
    result = cuDeviceGet(&device, 0);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: CUDA did not get a device." << std::endl;
    }

    int major, minor;
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA major version." << std::endl;
    }
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not get CUDA minor version." << std::endl;
    }

    std::cout << "Compute capability: " << major << "." << minor << std::endl;

    CUcontext context;
    result = cuCtxCreate(&context, 0, device);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not create CUDA context." << std::endl;
    }

    // CUfunction kernel = compileKernel(scaleCubes, "scaleCubes");
    CUfunction kernel = compileKernel2(scaleCubes, "scaleCubes");

    //const auto flatCache = carb::getCachedInterface<carb::flatcache::FlatCache>();
    // const auto iFabricUsd = carb::getCachedInterface<omni::fabric::IFabricUsd>();
    // auto framework = carb::getFramework();
    // p2a.platform.gpuCuda = framework->tryAcquireInterface<omni::gpucompute::GpuCompute>("omni.gpucompute-cuda.plugin");
    // p2a.platform.gpuCudaCtx = &p2a.platform.gpuCuda->createContext();


    //iterate over buckets but pass the vector for the whole bucket to the GPU.
    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        gsl::span<double> sizesD = fabricReaderWriter.getAttributeArrayGpu<double>(cubeBuckets, bucket, omni::fabric::Token("size"));

        double* ptr = sizesD.data();
        size_t elemCount = sizesD.size();
        void *args[] = { &ptr, &elemCount }; //NOLINT
        int blockSize, minGridSize;
        cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, nullptr, 0, 0);
        //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
        auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }

    result = cuCtxDestroy(context);
    if (result != CUDA_SUCCESS) {
        std::cout << "error: could not destroy CUDA context." << std::endl;
    }
}

CUfunction cesium::omniverse::FabricProceduralGeometry::compileKernel(const char *kernelSource, const char *kernelName) {

    CUfunction kernel_func;
    CUmodule module;

    // Set up JIT compilation options and compile the module
    // CUjit_option jitOptions[] = { //NOLINT
    //     CU_JIT_TARGET_FROM_CUCONTEXT,
    //     CU_JIT_OPTIMIZATION_LEVEL,
    // };

    // void *jitOptVals[] = { //NOLINT
    //     nullptr, // Target is picked from CUcontext
    //     (void *)3 // Optimization level 3
    // };

    const unsigned int jitNumOptions = 2;
    CUjit_option jitOptions[jitNumOptions]; //NOLINT
    void*        jitOptVals[jitNumOptions]; //NOLINT

    // Set up JIT compilation options to print verbose log
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = (void *)(size_t)jitLogBufferSize; //NOLINT

    char* jitLogBuffer = new char[jitLogBufferSize];
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER; //NOLINT
    jitOptVals[1] = jitLogBuffer; //NOLINT

    // Compile the module
    CUresult result = cuModuleLoadDataEx(&module, kernelSource, 2, jitOptions, jitOptVals);
    //attempt to use cuModuleLoadData instead of cuModuleLoadDataEx to see if it solves the CUDA_ERROR_INVALID_IMAGE (200) issue. (It did not.)
    //CUresult result = cuModuleLoadData(&module, kernelSource);

    std::cout << "JIT compilation log:\n" << jitLogBuffer << std::endl;
    delete[] jitLogBuffer;

    if (result != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(result, &error);
        std::cerr << "Could not compile module: " << error << std::endl;
        return nullptr;
    }

    // Get kernel function
    result = cuModuleGetFunction(&kernel_func, module, kernelName);
    if (result != CUDA_SUCCESS) {
        const char *error;
        cuGetErrorString(result, &error);
        std::cerr << "Could not get kernel function: " << error << std::endl;
        return nullptr;
    }

    return kernel_func;
}

CUfunction cesium::omniverse::FabricProceduralGeometry::compileKernel2(const char *kernelSource, const char *kernelName) {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernelSource, kernelName, 0, nullptr, nullptr);

    // Compile the program
    nvrtcCompileProgram(prog, 0, nullptr);

    // Get the PTX code
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    // Load the PTX code into a CUDA module
    CUmodule module;
    cuModuleLoadData(&module, ptx);

    // Get the kernel function from the module
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, kernelName);

    return kernel;
}

bool cesium::omniverse::FabricProceduralGeometry::checkCudaCompatibility() {
    int runtimeVer, driverVer;

    // Get CUDA Runtime version
    if (cudaRuntimeGetVersion(&runtimeVer) != cudaSuccess) {
        std::cerr << "Failed to get CUDA Runtime version" << std::endl;
        return false;
    }

    // Get CUDA driver version
    if (cudaDriverGetVersion(&driverVer) != cudaSuccess) {
        std::cerr << "Failed to get CUDA driver version" << std::endl;
        return false;
    }

    std::cout << "CUDA Runtime Version: " << runtimeVer / 1000 << "." << (runtimeVer % 100) / 10 << std::endl;
    std::cout << "CUDA Driver Version: " << driverVer / 1000 << "." << (driverVer % 100) / 10 << std::endl;

    // Check compatibility
    if (runtimeVer > driverVer) {
        std::cerr << "CUDA Toolkit version is greater than the driver version. "
                  << "Please update the CUDA driver." << std::endl;
        return false;
    }

    return true;
}


