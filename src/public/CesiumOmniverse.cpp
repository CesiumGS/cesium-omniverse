#define CARB_EXPORTS

#include "cesium/omniverse/CesiumOmniverse.h"

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUsdSchemas/data.h>
#include <carb/PluginUtils.h>
#include <carb/flatcache/FlatCache.h>

namespace cesium::omniverse {

class CesiumOmniversePlugin : public ICesiumOmniverseInterface {
  protected:
    void onStartup(const char* cesiumExtensionLocation) noexcept override {
        Context::onStartup(cesiumExtensionLocation);
    }

    void onShutdown() noexcept {
        Context::onShutdown();
    }

    void addCesiumDataIfNotExists(const char* token) noexcept override {
        CesiumIonClient::Token t;
        t.token = token;
        Context::instance().addCesiumDataIfNotExists(t);
    }

    int64_t addTilesetUrl(const char* url) noexcept override {
        return Context::instance().addTilesetUrl(url);
    }

    int64_t addTilesetIon(const char* name, int64_t ionId) noexcept override {
        const auto token = Context::instance().getDefaultToken();
        if (!token.has_value()) {
            return -1;
        }

        return addTilesetIon(name, ionId, token->token.c_str());
    }

    int64_t addTilesetIon(const char* name, int64_t ionId, const char* ionToken) noexcept override {
        return Context::instance().addTilesetIon(name, ionId, ionToken);
    }

    void addIonRasterOverlay(int64_t tilesetId, const char* name, int64_t ionId) noexcept override {
        const auto token = Context::instance().getDefaultToken();
        if (!token.has_value()) {
            return;
        }

        addIonRasterOverlay(tilesetId, name, ionId, token->token.c_str());
    }

    void
    addIonRasterOverlay(int64_t tilesetId, const char* name, int64_t ionId, const char* ionToken) noexcept override {
        Context::instance().addIonRasterOverlay(tilesetId, name, ionId, ionToken);
    }

    int64_t addTilesetAndRasterOverlay(
        const char* tilesetName,
        int64_t tilesetIonId,
        const char* rasterOverlayName,
        int64_t rasterOverlayIonId) noexcept override {
        const auto token = Context::instance().getDefaultToken();
        if (!token.has_value()) {
            return -1;
        }

        const auto tilesetId = addTilesetIon(tilesetName, tilesetIonId, token->token.c_str());
        addIonRasterOverlay(tilesetId, rasterOverlayName, rasterOverlayIonId, token->token.c_str());

        return tilesetId;
    }

    std::vector<std::pair<int64_t, const char*>> getAllTilesetIdsAndPaths() noexcept override {
        return Context::instance().getAllTilesetIdsAndPaths();
    }

    void removeTileset(int64_t tilesetId) noexcept override {
        Context::instance().removeTileset(tilesetId);
    }

    void reloadTileset(int64_t tilesetId) noexcept override {
        Context::instance().reloadTileset(tilesetId);
    }

    void onUpdateFrame(
        const pxr::GfMatrix4d& viewMatrix,
        const pxr::GfMatrix4d& projMatrix,
        double width,
        double height) noexcept override {
        const auto viewMatrixGlm = UsdUtil::usdToGlmMatrix(viewMatrix);
        const auto projMatrixGlm = UsdUtil::usdToGlmMatrix(projMatrix);
        Context::instance().onUpdateFrame(viewMatrixGlm, projMatrixGlm, width, height);
    }

    void onUpdateUi() noexcept override {
        Context::instance().onUpdateUi();
    }

    void onStageChange(long stageId) noexcept override {
        Context::onStageChange(stageId);
    }

    void setGeoreferenceOrigin(double longitude, double latitude, double height) noexcept override {
        CesiumGeospatial::Cartographic cartographic(glm::radians(longitude), glm::radians(latitude), height);
        Context::instance().setGeoreferenceOrigin(cartographic);
    }

    void connectToIon() noexcept override {
        Context::instance().connectToIon();
    }

    std::optional<std::shared_ptr<CesiumIonSession>> getSession() noexcept override {
        return Context::instance().getSession();
    }

    SetDefaultTokenResult getSetDefaultTokenResult() noexcept override {
        return Context::instance().getSetDefaultTokenResult();
    }

    bool isDefaultTokenSet() noexcept override {
        return Context::instance().isDefaultTokenSet();
    }

    void createToken(const char* name) noexcept override {
        Context::instance().createToken(name);
    }

    void selectToken(const char* id, const char* token) noexcept override {
        CesiumIonClient::Token t{id, "", token};
        Context::instance().selectToken(t);
    }

    void specifyToken(const char* token) noexcept override {
        Context::instance().specifyToken(token);
    }

    std::optional<AssetTroubleshootingDetails> getAssetTroubleshootingDetails() noexcept override {
        return Context::instance().getAssetTroubleshootingDetails();
    }

    std::optional<TokenTroubleshootingDetails> getAssetTokenTroubleshootingDetails() noexcept override {
        return Context::instance().getAssetTokenTroubleshootingDetails();
    }

    std::optional<TokenTroubleshootingDetails> getDefaultTokenTroubleshootingDetails() noexcept override {
        return Context::instance().getDefaultTokenTroubleshootingDetails();
    }

    void
    updateTroubleshootingDetails(int64_t tilesetId, uint64_t tokenEventId, uint64_t assetEventId) noexcept override {
        return Context::instance().updateTroubleshootingDetails(tilesetId, tokenEventId, assetEventId);
    }

    void updateTroubleshootingDetails(
        int64_t tilesetId,
        int64_t rasterOverlayId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept override {
        return Context::instance().updateTroubleshootingDetails(tilesetId, rasterOverlayId, tokenEventId, assetEventId);
    }

    std::string printFabricStage() noexcept override {
        return FabricUtil::printFabricStage();
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
CARB_PLUGIN_IMPL_DEPS(carb::flatcache::FlatCache, carb::flatcache::IStageInProgress)

void fillInterface([[maybe_unused]] cesium::omniverse::CesiumOmniversePlugin& iface) {}
