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
std::unordered_map<int64_t, std::shared_ptr<OmniTileset>> tilesets;
std::optional<TokenTroubleshootingDetails> tokenTroubleshootingDetails = std::nullopt;
std::optional<AssetTroubleshootingDetails> assetTroubleshootingDetails = std::nullopt;
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

    int64_t addTilesetIon(const char* name, int64_t ionId) noexcept override {
        auto token = OmniTileset::getDefaultToken();
        if (!token.has_value()) {
            return -1;
        }

        return addTilesetIon(name, ionId, token->token.c_str());
    }

    int64_t addTilesetIon([[maybe_unused]] const char* name, int64_t ionId, const char* ionToken) noexcept override {
        tilesets.insert({ionId, std::make_unique<OmniTileset>(ionId, ionToken)});
        return ionId;
    }

    int64_t addTilesetAndRasterOverlay(
        const char* tilesetName,
        int64_t tilesetIonId,
        const char* rasterOverlayName,
        int64_t rasterOverlayIonId) noexcept override {
        auto token = OmniTileset::getDefaultToken();
        if (!token.has_value()) {
            return -1;
        }

        auto id = addTilesetIon(tilesetName, tilesetIonId, token->token.c_str());
        addIonRasterOverlay(id, rasterOverlayName, rasterOverlayIonId, token->token.c_str());

        return id;
    }

    std::vector<std::pair<int64_t, const char*>> getAllTilesetIdsAndPaths() noexcept override {
        if (tilesets.empty()) {
            return {};
        }

        std::vector<std::pair<int64_t, const char*>> result(tilesets.size());
        for (const auto& item : tilesets) {
            auto path = item.second->getPath();
            result.emplace_back(item.first, path.GetString().c_str());
        }

        return result;
    }

    void removeTileset(int tileset) noexcept override {
        tilesets.erase(tileset);
    }

    void addIonRasterOverlay(int64_t tileset, const char* name, int64_t ionId) noexcept override {
        auto token = OmniTileset::getDefaultToken();
        if (!token.has_value()) {
            return;
        }

        addIonRasterOverlay(tileset, name, ionId, token->token.c_str());
    }

    void addIonRasterOverlay(int64_t tileset, const char* name, int64_t ionId, const char* ionToken) noexcept override {
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

    std::optional<TokenTroubleshootingDetails> getTokenTroubleshootingDetails() noexcept override {
        return tokenTroubleshootingDetails;
    }

    std::optional<AssetTroubleshootingDetails> getAssetTroubleshootingDetails() noexcept override {
        return assetTroubleshootingDetails;
    }

    void
    updateTroubleshootingDetails(int64_t tilesetId, uint64_t tokenEventId, uint64_t assetEventId) noexcept override {
        auto tileset = tilesets.find(tilesetId);

        if (tileset == tilesets.end()) {
            return;
        }

        tokenTroubleshootingDetails = TokenTroubleshootingDetails();
        assetTroubleshootingDetails = AssetTroubleshootingDetails();

        TokenTroubleshooter troubleshooter(tileset->second);
        troubleshooter.updateTokenTroubleshootingDetails(tilesetId, tokenEventId, tokenTroubleshootingDetails.value());
        troubleshooter.updateAssetTroubleshootingDetails(tilesetId, assetEventId, assetTroubleshootingDetails.value());
    }

    void updateTroubleshootingDetails(
        int64_t tilesetId,
        int64_t rasterOverlayId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept override {
        auto tileset = tilesets.find(tilesetId);

        if (tileset == tilesets.end()) {
            return;
        }

        tokenTroubleshootingDetails = TokenTroubleshootingDetails();
        assetTroubleshootingDetails = AssetTroubleshootingDetails();

        TokenTroubleshooter troubleshooter(tileset->second);
        troubleshooter.updateTokenTroubleshootingDetails(
            rasterOverlayId, tokenEventId, tokenTroubleshootingDetails.value());
        troubleshooter.updateAssetTroubleshootingDetails(
            rasterOverlayId, assetEventId, assetTroubleshootingDetails.value());
    }

    void onUiUpdate() noexcept override {
        OmniTileset::onUiUpdate();
    }

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
