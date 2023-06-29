#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/flatcache/FlatCache.h>
#include <carb/flatcache/FlatCacheUSD.h>
// #include <omni/usd/omni.h>
// omni/usd/UsdUtils.h
#include <omni/usd/UsdContextIncludes.h>
#include <omni/usd/UsdContext.h>
#include <pxr/usd/usd/prim.h>


int cesium::omniverse::FabricProceduralGeometry::createCube() {
    // auto stage = omni::usd::get_context()->get_stage();

    // // Set the USD stage
    // _stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

    // // Set the Fabric stage
    // const auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    // const auto stageInProgressId =
    //     iStageInProgress->get(carb::flatcache::UsdStageId{static_cast<uint64_t>(stageId)});
    // _fabricStageInProgress = carb::flatcache::StageInProgress(stageInProgressId);


    const pxr::UsdStageRefPtr usdStagePtr = Context::instance().getStage();
    // auto usdStagePtr = omni::usd::UsdContext::getContext()->getStage();

    // why can't we access GetStageId()?
    // carb::flatcache::UsdStageId usdStageId = usdStageRefPtr->GetStageId();
    long id = Context::instance().getStageId();
    auto usdStageId = carb::flatcache::UsdStageId{static_cast<uint64_t>(id)};

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
    double size = *sizePtr;
    size = size * 10;

    // //write our changes back to USD.
    // carb::flatcache::FlatCache *iFabric;
    // iFabric->cacheToUsd(*iFabric->getCache(usdStageId, flatcache::kDefaultUserId));

    std::shared_ptr<carb::flatcache::FlatCache> flatCache = std::make_shared<carb::flatcache::FlatCache>();
    if (flatCache == nullptr) {
        return -1;
    }
    //iFabric->cacheToUsd(*iFabric->getCache(usdStageId, flatcache::kDefaultUserId));
    auto& pathToAttributesMap = flatCache->createCache(usdStageId, flatcache::kDefaultUserId, carb::flatcache::CacheType::eWithoutHistory);
    //auto pathToAttributesMapPtr = flatCache->getCache(usdStageId, flatcache::kDefaultUserId);
    // auto pam = *pamPtr;
    // flatCache->cacheToUsd(pam);
    //flatCache->cacheToUsd(*pathToAttributesMapPtr);
    flatCache->cacheToUsd(pathToAttributesMap);

    // //check that Fabric correctly modified the USD stage.
    pxr::UsdAttribute sizeAttr = prim.GetAttribute(pxr::TfToken("size"));
    double value;
    sizeAttr.Get(&value);
    if (value == 10.f) {
        value = 20;
    } else {
        value = 5.f;
    }
    // CHECK(value == 10.0f);

    return 178;
}


//auto stageInProgress = UsdUtil::getFabricStageInProgress();
