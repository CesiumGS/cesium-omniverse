#pragma once

#include "CesiumIonSession.h"

#include "cesium/omniverse/Georeference.h"
#include "cesium/omniverse/RenderResourcesPreparer.h"
#include "cesium/omniverse/SetDefaultTokenResult.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IonRasterOverlay.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/stageCache.h>
#include <pxr/usd/usdUtils/stageCache.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace cesium::omniverse {
class OmniTileset {
  public:
    OmniTileset(const std::string& url);

    OmniTileset(int64_t ionID, const std::string& ionToken);

    void updateFrame(const pxr::GfMatrix4d& viewMatrix, const pxr::GfMatrix4d& projMatrix, double width, double height);

    void addIonRasterOverlay(const std::string& name, int64_t ionId, const std::string& ionToken);

    static void init(const std::filesystem::path& cesiumExtensionLocation);

    static pxr::UsdStageRefPtr& getStage();

    static void setStage(const pxr::UsdStageRefPtr& stage);

    static void connectToIon();

    static void addCesiumDataIfNotExists(const CesiumIonClient::Token& token);

    static SetDefaultTokenResult getSetDefaultTokenResult();

    static void createToken(const std::string& name);

    static void selectToken(const CesiumIonClient::Token& token);

    static void specifyToken(const std::string& token);

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
