#include "cesium/omniverse/FabricProceduralGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/flatcache/FlatCacheUSD.h>
// #include <omni/usd/omni.h>
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

    const pxr::UsdStageRefPtr usdStageRefPtr = Context::instance().getStage();

    // why can't we access GetStageId()?
    // carb::flatcache::UsdStageId usdStageId = usdStageRefPtr->GetStageId();
    long usdStageId = Context::instance().getStageId();

    //create a cube in USD and set its size.
    pxr::UsdPrim prim = usdStageRefPtr->DefinePrim(pxr::SdfPath("/TestCube"), pxr::TfToken("Cube"));
    prim.CreateAttribute(pxr::TfToken("size"), pxr::SdfValueTypeNames->Double).Set(1.0);

    //prefetch it to Fabric’s cache.
    carb::flatcache::Path primPath("/TestCube");
    // carb::flatcache::StageInProgress stageInProgress = UsdUtil::getFabricStageInProgress();

    //see UsdValueAccessors.h
    auto iStageInProgress = carb::getCachedInterface<carb::flatcache::IStageInProgress>();
    auto id = carb::flatcache::UsdStageId{static_cast<uint64_t>(usdStageId)};
    // auto id = carb::flatcache::UsdStageId{usdStageId};
    iStageInProgress->prefetchPrim(id, primPath);
    // auto stageInProgress = iStageInProgress->get(usdStageId);

    // stageInProgress.prefetchPrim(usdStageId, primPath);

    // //use Fabric to modify the cube’s dimensions
    // double& size = *stage.getAttribute<double>(primPath, Token("size"));
    // size = size * 10;

    // //write our changes back to USD.
    // iFabric->cacheToUsd(*iFabric->getCache(usdStageId, flatcache::kDefaultUserId));

    // //check that Fabric correctly modified the USD stage.
    // pxr::UsdAttribute sizeAttr = prim.GetAttribute(pxr::TfToken("size"));
    // double value;
    // sizeAttr.Get(&value);
    // CHECK(value == 10.0f);

    return 178;
}


//auto stageInProgress = UsdUtil::getFabricStageInProgress();
