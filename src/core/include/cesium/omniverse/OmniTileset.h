#pragma once

#include "CesiumIonSession.h"

#include "cesium/omniverse/Georeference.h"
#include "cesium/omniverse/RenderResourcesPreparer.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IonRasterOverlay.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <pxr/usd/usd/stage.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace cesium::omniverse {
class OmniTileset {
  public:
    OmniTileset(const pxr::UsdStageRefPtr& stage, const std::string& url);

    OmniTileset(const pxr::UsdStageRefPtr& stage, int64_t ionID, const std::string& ionToken);

    void updateFrame(const pxr::GfMatrix4d& viewMatrix, const pxr::GfMatrix4d& projMatrix, double width, double height);

    void addIonRasterOverlay(const std::string& name, int64_t ionId, const std::string& ionToken);

    static void init(const std::filesystem::path& cesiumExtensionLocation);

    static void connectToIon();

    static void onUiUpdate();

    static std::optional<std::shared_ptr<CesiumIonSession>> getSession();

    static void shutdown();

  private:
    void initOriginShiftHandler();

    std::vector<Cesium3DTilesSelection::ViewState> viewStates;
    std::shared_ptr<RenderResourcesPreparer> renderResourcesPreparer;
    std::unique_ptr<Cesium3DTilesSelection::Tileset> tileset;
    OriginShiftHandler originShiftHandler;
    CesiumUtility::IntrusivePointer<Cesium3DTilesSelection::IonRasterOverlay> rasterOverlay;
};
} // namespace cesium::omniverse
