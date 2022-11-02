#pragma once

#include "Georeference.h"
#include "RenderResourcesPreparer.h"

#include "cesium/omniverse/CesiumOmniverse.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/Tileset.h>

#ifdef CESIUM_OMNI_GCC
#define _GLIBCXX_PERMIT_BACKWARD_HASH
#endif

#include <pxr/usd/usd/stage.h>

#include "UtilMacros.h"

#include <memory>
#include <vector>

namespace Cesium {
class CESIUM_OMNI_EXPORT_CLASS OmniTileset {
  public:
    OmniTileset(const pxr::UsdStageRefPtr& stage, const std::string& url);

    OmniTileset(const pxr::UsdStageRefPtr& stage, int64_t ionID, const std::string& ionToken);

    void updateFrame(const pxr::GfMatrix4d& viewMatrix, const pxr::GfMatrix4d& projMatrix, double width, double height);

    static void init();

    static void shutdown();

  private:
    void initOriginShiftHandler();

    std::vector<Cesium3DTilesSelection::ViewState> viewStates;
    std::shared_ptr<RenderResourcesPreparer> renderResourcesPreparer;
    std::unique_ptr<Cesium3DTilesSelection::Tileset> tileset;
    OriginShiftHandler originShiftHandler;
};
} // namespace Cesium
