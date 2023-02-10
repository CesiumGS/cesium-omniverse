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

    void addCesiumDataIfNotExists(const char* token) noexcept override {
        CesiumIonClient::Token t;
        t.token = token;
        OmniTileset::addCesiumDataIfNotExists(t);
    }

    int addTilesetUrl(const char* url) noexcept override {
        const int tilesetId = currentId++;
        tilesets.insert({tilesetId, std::make_unique<OmniTileset>(url)});
        return tilesetId;
    }

    int addTilesetIon(const char* name, int64_t ionId) noexcept override {
        auto token = OmniTileset::getDefaultToken();
        if (!token.has_value()) {
            return -1;
        }

        return addTilesetIon(name, ionId, token->token.c_str());
    }

    int addTilesetIon([[maybe_unused]] const char* name, int64_t ionId, const char* ionToken) noexcept override {
        const int tilesetId = currentId++;
        tilesets.insert({tilesetId, std::make_unique<OmniTileset>(ionId, ionToken)});
        return tilesetId;
    }

    int addTilesetAndRasterOverlay(
        const char* tilesetName,
        int64_t tilesetIonId,
        const char* rasterOverlayName,
        int64_t rasterOverlayIonId) noexcept override {
        if (!OmniTileset::getSession().has_value()) {
            return -1;
        }

        auto token = OmniTileset::getDefaultToken();
        if (!token.has_value()) {
            return -1;
        }

        auto id = addTilesetIon(tilesetName, tilesetIonId, token->token.c_str());
        addIonRasterOverlay(id, rasterOverlayName, rasterOverlayIonId, token->token.c_str());

        return id;
    }

    void removeTileset(int tileset) noexcept override {
        tilesets.erase(tileset);
    }

    void addIonRasterOverlay(int tileset, const char* name, int64_t ionId) noexcept override {
        auto token = OmniTileset::getDefaultToken();
        if (!token.has_value()) {
            return;
        }

        addIonRasterOverlay(tileset, name, ionId, token->token.c_str());
    }

    void addIonRasterOverlay(int tileset, const char* name, int64_t ionId, const char* ionToken) noexcept override {
        const auto iter = tilesets.find(tileset);
        if (iter != tilesets.end()) {
            iter->second->addIonRasterOverlay(name, ionId, ionToken);
        }
    }

    void updateFrame(
        const pxr::GfMatrix4d& viewMatrix,
        const pxr::GfMatrix4d& projMatrix,
        double width,
        double height) noexcept override {
        for (const auto& tileset : tilesets) {
            tileset.second->updateFrame(viewMatrix, projMatrix, width, height);
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

    SetDefaultTokenResult getSetDefaultTokenResult() noexcept override {
        return OmniTileset::getSetDefaultTokenResult();
    }

    bool isDefaultTokenSet() noexcept override {
        auto token = OmniTileset::getDefaultToken();

        return token.has_value();
    }

    void createToken(const char* name) noexcept override {
        OmniTileset::createToken(name);
    }

    void selectToken(const char* id, const char* token) noexcept override {
        CesiumIonClient::Token t{id, "", token};
        OmniTileset::selectToken(t);
    }

    void specifyToken(const char* token) noexcept override {
        OmniTileset::specifyToken(token);
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
