#define CARB_EXPORTS

#include "cesium/omniverse/CesiumOmniverse.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/SessionRegistry.h"
#include "cesium/omniverse/UsdUtil.h"
#include "cesium/omniverse/Viewport.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUsdSchemas/data.h>
#include <carb/PluginUtils.h>
#include <omni/fabric/IFabric.h>

namespace cesium::omniverse {

class CesiumOmniversePlugin final : public ICesiumOmniverseInterface {
  protected:
    void onStartup(const char* cesiumExtensionLocation) noexcept override {
        Context::onStartup(cesiumExtensionLocation);
    }

    void onShutdown() noexcept override {
        Context::onShutdown();
    }

    void reloadTileset(const char* tilesetPath) noexcept override {
        Context::instance().reloadTileset(pxr::SdfPath(tilesetPath));
    }

    void onUpdateFrame(const std::vector<Viewport>& viewports) noexcept override {
        Context::instance().onUpdateFrame(viewports);
    }

    void onUpdateUi() noexcept override {
        Context::instance().onUpdateUi();
    }

    void onStageChange(long stageId) noexcept override {
        Context::instance().setStageId(stageId);
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

    std::string getServerPath() noexcept override {
        return UsdUtil::getPathToCurrentIonServer().GetString();
    }

    std::vector<std::shared_ptr<CesiumIonSession>> getAllSessions() noexcept override {
        return Context::instance().getAllSessions();
    }

    std::vector<std::string> getAllServerPaths() noexcept override {
        const auto paths = SessionRegistry::getInstance().getAllServerPaths();

        std::vector<std::string> result;
        result.reserve(paths.size());

        for (const auto& path : paths) {
            result.emplace_back(path.GetString());
        }

        return result;
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

    void updateTroubleshootingDetails(
        const char* tilesetPath,
        int64_t tilesetIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept override {
        return Context::instance().updateTroubleshootingDetails(
            pxr::SdfPath(tilesetPath), tilesetIonAssetId, tokenEventId, assetEventId);
    }

    void updateTroubleshootingDetails(
        const char* tilesetPath,
        int64_t tilesetIonAssetId,
        int64_t imageryIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept override {
        return Context::instance().updateTroubleshootingDetails(
            pxr::SdfPath(tilesetPath), tilesetIonAssetId, imageryIonAssetId, tokenEventId, assetEventId);
    }

    std::string printFabricStage() noexcept override {
        return FabricUtil::printFabricStage();
    }

    RenderStatistics getRenderStatistics() noexcept override {
        return Context::instance().getRenderStatistics();
    }

    bool creditsAvailable() noexcept override {
        return Context::instance().creditsAvailable();
    }

    std::vector<std::pair<std::string, bool>> getCredits() noexcept override {
        return Context::instance().getCredits();
    }

    void creditsStartNextFrame() noexcept override {
        return Context::instance().creditsStartNextFrame();
    }

    void addGlobeAnchorToPrim(const char* path) noexcept override {
        return Context::instance().addGlobeAnchorToPrim(pxr::SdfPath(path));
    }

    void addGlobeAnchorToPrim(const char* path, double latitude, double longitude, double height) noexcept override {
        return Context::instance().addGlobeAnchorToPrim(pxr::SdfPath(path), latitude, longitude, height);
    }

    void addCartographicPolygonPrim(const char* path) noexcept override {
        Context::instance().addCartographicPolygonPrim(pxr::SdfPath(path));
    }

    bool isTracingEnabled() noexcept override {
#if CESIUM_TRACING_ENABLED
        return true;
#else
        return false;
#endif
    }
};
} // namespace cesium::omniverse

const struct carb::PluginImplDesc pluginImplDesc = {
    "cesium.omniverse.plugin",
    "Cesium Omniverse Carbonite Plugin.",
    "Cesium",
    carb::PluginHotReload::eDisabled,
    "dev"};

// NOLINTBEGIN
CARB_PLUGIN_IMPL(pluginImplDesc, cesium::omniverse::CesiumOmniversePlugin)
CARB_PLUGIN_IMPL_DEPS(omni::fabric::IFabric, omni::fabric::IStageReaderWriter)
// NOLINTEND

void fillInterface([[maybe_unused]] cesium::omniverse::CesiumOmniversePlugin& iface) {}
