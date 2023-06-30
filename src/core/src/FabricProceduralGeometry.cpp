#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/flatcache/FlatCache.h>
// #include <carb/flatcache/FlatCacheUSD.h>
// #include <omni/usd/omni.h>
// #include <omni/usd/UsdContextIncludes.h>
// #include <omni/usd/UsdContext.h>

// #include <CesiumUsdSchemas/data.h>
#include "pxr/base/tf/token.h"

#include <pxr/usd/usd/prim.h>
#include <omni/gpucompute/GpuCompute.h>


int cesium::omniverse::FabricProceduralGeometry::createCube() {
    modifyUsdPrim();
    //modify1000Prims();
    //modify1000PrimsViaCuda();

    return 178;
}

void cesium::omniverse::FabricProceduralGeometry::modifyUsdPrim() {
    //Linker error getting UsdContext using omni::usd
    // auto context = omni::usd::UsdContext::getContext();
    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();

    long id = Context::instance().getStageId();
    auto usdStageId = carb::flatcache::UsdStageId{static_cast<uint64_t>(id)};

    //create a cube in USD and set its size.
    pxr::UsdPrim prim = usdStagePtr->DefinePrim(pxr::SdfPath("/TestCube"), pxr::TfToken("Cube"));
    prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(1.0);

    //prefetch it to Fabric’s cache.
    auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    carb::flatcache::Path primPath("/TestCube");
    iStageInProgress->prefetchPrim(usdStageId, primPath);

    //use Fabric to modify the cube’s dimensions
    carb::flatcache::StageInProgress sip = UsdUtil::getFabricStageInProgress();
    auto sizeToken = carb::flatcache::Token("size");
    auto* sizePtr = sip.getAttribute<double>(primPath, sizeToken);
    *sizePtr *= 10;

    // //write our changes back to USD.
    const auto flatCache = carb::getCachedInterface<carb::flatcache::FlatCache>();
    if (flatCache->createCache != nullptr) {
        auto& pathToAttributesMap = flatCache->createCache(usdStageId, flatcache::kDefaultUserId, carb::flatcache::CacheType::eWithoutHistory);
        //auto pathToAttributesMapPtr = flatCache->getCache(usdStageId, flatcache::kDefaultUserId);
        // auto pam = *pamPtr;
        // flatCache->cacheToUsd(pam);
        //flatCache->cacheToUsd(*pathToAttributesMapPtr);
        flatCache->cacheToUsd(pathToAttributesMap);
    }

    // //check that Fabric correctly modified the USD stage.
//    pxr::UsdAttribute sizeAttr = prim.GetAttribute(pxr::TfToken("size"));
//    double value;
//    sizeAttr.Get(&value);
//    if (value == 10.f) {
//        value = 20;
//    } else {
//        value = 5.f;
//    }
    // CHECK(value == 10.0f);
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
    auto usdStageId = carb::flatcache::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        carb::flatcache::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageInProgress->prefetchPrim(usdStageId, path);
    }

    //tell Fabric which prims to change. Select all prims of type Cube.
    carb::flatcache::AttrNameAndType cubeTag(
        carb::flatcache::Type(carb::flatcache::BaseDataType::eTag, 1, 0, carb::flatcache::AttributeRole::ePrimTypeName),
        carb::flatcache::Token("Cube"));

    const auto stageInProgressId = iStageInProgress->get(carb::flatcache::UsdStageId{static_cast<uint64_t>(id)});
    auto fabricStageInProgress = carb::flatcache::StageInProgress(stageInProgressId);
    carb::flatcache::PrimBucketList cubeBuckets = fabricStageInProgress.findPrims({ cubeTag });
    //carb::flatcache::PrimBucketList cubeBuckets = iStageInProgress->findPrims({ cubeTag });
    //carb::flatcache::PrimBucketList cubeBuckets = stage.findPrims({ cubeTag });

    // Fabric is free to store the 1000 cubes in as many buckets as it likes...iterate over the buckets
    for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    {
        auto sizes = fabricStageInProgress.getAttributeArray<double>(cubeBuckets, bucket, carb::flatcache::Token("size"));
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
    auto usdStageId = carb::flatcache::UsdStageId{static_cast<uint64_t>(id)};
    auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    for (size_t i = 0; i != cubeCount; i++)
    {
        carb::flatcache::Path path(("/cube_" + std::to_string(i)).c_str());
        iStageInProgress->prefetchPrim(usdStageId, path);
    }

    carb::flatcache::AttrNameAndType cubeTag(carb::flatcache::Type(carb::flatcache::BaseDataType::eTag, 1, 0, carb::flatcache::AttributeRole::ePrimTypeName), carb::flatcache::Token("Cube"));
    const auto stageInProgressId = iStageInProgress->get(carb::flatcache::UsdStageId{static_cast<uint64_t>(id)});
    auto fabricStageInProgress = carb::flatcache::StageInProgress(stageInProgressId);
    carb::flatcache::PrimBucketList cubeBuckets = fabricStageInProgress.findPrims({ cubeTag });

    //CUDA via CUDA_JIT and string
    // static const char* scaleCubes =
    //     "   extern \"C\" __global__"
    //     "   void scaleCubes(double* cubeSizes, size_t count)"
    //     "   {"
    //     "       size_t i = blockIdx.x * blockDim.x + threadIdx.x;"
    //     "       if(count<=i) return;"
    //     ""
    //     "       cubeSizes[i] *= 10.0;"
    //     "   }";

    // const auto flatCache = carb::getCachedInterface<carb::flatcache::FlatCache>();
    // carb::flatcache::PathToAttributesMap& p2a = flatCache->getCache(usdStageId, flatcache::kDefaultUserId);
    // p2a.platform.gpuCuda = framework->tryAcquireInterface<omni::gpucompute::GpuCompute>("omni.gpucompute-cuda.plugin");
    // p2a.platform.gpuCudaCtx = &p2a.platform.gpuCuda->createContext();
    // CUfunction kernel = compileKernel(scaleCubes, "scaleCubes");

    // //iterate over buckets but pass the vector for the whole bucket to the GPU.
    // for (size_t bucket = 0; bucket != cubeBuckets.bucketCount(); bucket++)
    // {
    //     gsl::span<double> sizesD = fabricStageInProgress.getAttributeArrayGpu<double>(cubeBuckets, bucket, carb::flatcache::Token("size"));

    //     double* ptr = sizesD.data();
    //     size_t elemCount = sizesD.size();
    //     void *args[] = { &ptr, &elemCount };
    //     int blockSize, minGridSize;
    //     cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, nullptr, 0, 0);
    //     //CUresult err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
    //     auto err = cuLaunchKernel(kernel, minGridSize, 1, 1, blockSize, 1, 1, 0, NULL, args, 0);
    //     // REQUIRE(!err);
    //     if (err) {
    //         return;
    //     }
    // }

}
