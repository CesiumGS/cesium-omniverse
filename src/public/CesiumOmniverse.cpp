#define CARB_EXPORTS

#include "cesium/omniverse/CesiumOmniverse.h"

#include "CesiumUsdSchemas/data.h"

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/OmniTileset.h"

#include <carb/PluginUtils.h>
#include <pxr/usd/usd/stageCache.h>
#include <pxr/usd/usdUtils/stageCache.h>

#include <unordered_map>

namespace cesium::omniverse {

namespace {
int currentId = 0;
std::unordered_map<int, std::unique_ptr<OmniTileset>> tilesets;
} // namespace

class CesiumOmniversePlugin : public ICesiumOmniverseInterface {
  protected:
    void initialize(const char* cesiumMemLocation) noexcept override {
        OmniTileset::init(cesiumMemLocation);
    }

    void finalize() noexcept {
        OmniTileset::shutdown();
    }

    void addCesiumData(long stageId, const char* ionToken) noexcept override {
        const auto& stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));
        pxr::UsdPrim cesiumDataPrim = stage->DefinePrim(pxr::SdfPath("/Cesium"));
        pxr::CesiumData cesiumData(cesiumDataPrim);
        auto defaultTokenId = cesiumData.CreateDefaultProjectTokenIdAttr(pxr::VtValue(""));
        auto defaultToken = cesiumData.CreateDefaultProjectTokenAttr(pxr::VtValue(""));

        if (strlen(ionToken) != 0) {
            defaultTokenId.Set("");
            defaultToken.Set(ionToken);
        }
    }

    int addTilesetUrl(const char* url) noexcept override {
        const int tilesetId = currentId++;
        tilesets.insert({tilesetId, std::make_unique<OmniTileset>(url)});
        return tilesetId;
    }

    int addTilesetIon(int64_t ionId, const char* ionToken) noexcept override {
        const int tilesetId = currentId++;
        tilesets.insert({tilesetId, std::make_unique<OmniTileset>(ionId, ionToken)});
        return tilesetId;
    }

    void removeTileset(int tileset) noexcept override {
        tilesets.erase(tileset);
    }

    void addIonRasterOverlay(int tileset, const char* name, int64_t ionId, const char* ionToken) noexcept override {
        const auto iter = tilesets.find(tileset);
        if (iter != tilesets.end()) {
            iter->second->addIonRasterOverlay(name, ionId, ionToken);
        }
    }

    void updateFrame(
        int tileset,
        const pxr::GfMatrix4d& viewMatrix,
        const pxr::GfMatrix4d& projMatrix,
        double width,
        double height) noexcept override {
        const auto iter = tilesets.find(tileset);
        if (iter != tilesets.end()) {
            iter->second->updateFrame(viewMatrix, projMatrix, width, height);
        }
    }

    void updateStage(long stageId) noexcept override {
        const auto& stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));
        OmniTileset::setStage(stage);
    }

    void setGeoreferenceOrigin(double longitude, double latitude, double height) noexcept override {
        Georeference::instance().setOrigin(CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(
            CesiumGeospatial::Cartographic(glm::radians(longitude), glm::radians(latitude), height)));
    }

    void connectToIon() noexcept override {
        OmniTileset::connectToIon();
    }

    void onUiUpdate() noexcept override {
        OmniTileset::onUiUpdate();
    };

    std::optional<std::shared_ptr<CesiumIonSession>> getSession() noexcept override {
        return OmniTileset::getSession();
    }
};

} // namespace cesium::omniverse

const struct carb::PluginImplDesc pluginImplDesc = {
    "cesium.omniverse.plugin",
    "Cesium Omniverse Carbonite Plugin.",
    "Cesium",
    carb::PluginHotReload::eDisabled,
    "dev"};

CARB_PLUGIN_IMPL(pluginImplDesc, cesium::omniverse::CesiumOmniversePlugin)

void fillInterface([[maybe_unused]] cesium::omniverse::CesiumOmniversePlugin& iface) {}
