#pragma once

#include "cesium/omniverse/Georeference.h"
#include "cesium/omniverse/RenderResourcesPreparer.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/Tileset.h>
#include <pxr/usd/usd/stage.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace Cesium {
class OmniTileset {
  public:
    OmniTileset(const pxr::UsdStageRefPtr& stage, const std::string& url);

    OmniTileset(const pxr::UsdStageRefPtr& stage, int64_t ionID, const std::string& ionToken);

    void updateFrame(const pxr::GfMatrix4d& viewMatrix, const pxr::GfMatrix4d& projMatrix, double width, double height);

    static void init(const std::filesystem::path& cesiumMemLocation);

    static void shutdown();

  private:
    void initOriginShiftHandler();

    std::vector<Cesium3DTilesSelection::ViewState> viewStates;
    std::shared_ptr<RenderResourcesPreparer> renderResourcesPreparer;
    std::unique_ptr<Cesium3DTilesSelection::Tileset> tileset;
    OriginShiftHandler originShiftHandler;
};
} // namespace Cesium
