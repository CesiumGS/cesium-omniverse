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

    CUfunction kernel = compileKernel(scaleCubes, "scaleCubes");

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
        auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
        // REQUIRE(!err);
        if (err) {
            std::cout << "error" << std::endl;
        }
    }
}

CUfunction cesium::omniverse::FabricProceduralGeometry::compileKernel(const char *kernelSource, const char *kernelName) {
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kernelSource, "myKernel.cu", 0, nullptr, nullptr);

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
