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


int cesium::omniverse::FabricProceduralGeometry::createCube() {
    //modifyUsdPrim();
    modify1000Prims();

    return 178;
}

void cesium::omniverse::FabricProceduralGeometry::modifyUsdPrim() {
    // // Set the USD stage
    // _stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

    // // Set the Fabric stage
    // const auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    // const auto stageInProgressId =
    //     iStageInProgress->get(carb::flatcache::UsdStageId{static_cast<uint64_t>(stageId)});
    // _fabricStageInProgress = carb::flatcache::StageInProgress(stageInProgressId);


    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    // linker error:
    // const pxr::UsdStageWeakPtr usdStagePtr = omni::usd::UsdContext::getContext()->getStage();

    //can you get the Context via omni::usd::UsdContext::getContext()
    // auto context = omni::usd::UsdContext::getContext();
    // if (context == nullptr) {
    //     return;
    // }

    long id = Context::instance().getStageId();
    // ported from Python omni.usd.get_context().get_stage_id(), but linker error:
    // auto id2 = omni::usd::UsdContext::getContext()->getStageId();
    auto usdStageId = carb::flatcache::UsdStageId{static_cast<uint64_t>(id)};
    // auto id = omni::usd::UsdContext::getContext()->getStageId();

    //create a cube in USD and set its size.
    pxr::UsdPrim prim = usdStagePtr->DefinePrim(pxr::SdfPath("/TestCube"), pxr::TfToken("Cube"));
    prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(1.0);

    //prefetch it to Fabric’s cache.
    carb::flatcache::Path primPath("/TestCube");
    // carb::flatcache::StageInProgress stageInProgress = UsdUtil::getFabricStageInProgress();

    //see UsdValueAccessors.h
    auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    // auto stageInProgressId = iStageInProgress->get(usdStageId);

    iStageInProgress->prefetchPrim(usdStageId, primPath);

    //use Fabric to modify the cube’s dimensions
    carb::flatcache::StageInProgress sip = UsdUtil::getFabricStageInProgress();
    auto s = carb::flatcache::Token("size");
    auto* sizePtr = sip.getAttribute<double>(primPath, s);
    *sizePtr *= 10;
    // double size = *sizePtr;
    // size = size * 10;

    // //write our changes back to USD.
    // iFabric->cacheToUsd(*iFabric->getCache(usdStageId, flatcache::kDefaultUserId));
    std::shared_ptr<carb::flatcache::FlatCache> flatCache = std::make_shared<carb::flatcache::FlatCache>();
    //iFabric->cacheToUsd(*iFabric->getCache(usdStageId, flatcache::kDefaultUserId));
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
