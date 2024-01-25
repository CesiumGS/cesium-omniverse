#define CARB_EXPORTS

#include "cesium/omniverse/CesiumOmniverse.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/CesiumIonServerManager.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/OmniData.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"
#include "cesium/omniverse/Viewport.h"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumUtility/CreditSystem.h>
#include <carb/PluginUtils.h>
#include <omni/fabric/IFabric.h>
#include <omni/kit/IApp.h>

#include <gsl/span>

namespace cesium::omniverse {

class CesiumOmniversePlugin final : public ICesiumOmniverseInterface {
  protected:
    void onStartup(const char* cesiumExtensionLocation) noexcept override {
        _pContext = std::make_unique<Context>(cesiumExtensionLocation);
    }

    void onShutdown() noexcept override {
        _pContext = nullptr;
    }

    void reloadTileset(const char* tilesetPath) noexcept override {
        const auto pTileset = _pContext->getAssetRegistry().getTileset(pxr::SdfPath(tilesetPath));

        if (pTileset) {
            pTileset->reload();
        }
    }

    void onUpdateFrame(const ViewportApi* viewports, uint64_t count) noexcept override {
        const auto span = gsl::span<const Viewport>(reinterpret_cast<const Viewport*>(viewports), count);
        _pContext->onUpdateFrame(span);
    }

    void onUsdStageChanged(long stageId) noexcept override {
        _pContext->onUsdStageChanged(stageId);
    }

    void connectToIon() noexcept override {
        _pContext->getCesiumIonServerManager().connectToIon();
    }

    std::optional<std::shared_ptr<CesiumIonSession>> getSession() noexcept override {
        return _pContext->getCesiumIonServerManager().getCurrentIonSession();
    }

    std::string getServerPath() noexcept override {
        const auto pIonServer = _pContext->getCesiumIonServerManager().getCurrentIonServer();

        if (!pIonServer) {
            return "";
        }

        return pIonServer->getPath().GetString();
    }

    std::vector<std::shared_ptr<CesiumIonSession>> getSessions() noexcept override {
        const auto& ionServers = _pContext->getAssetRegistry().getIonServers();

        std::vector<std::shared_ptr<CesiumIonSession>> result;
        result.reserve(ionServers.size());

        for (const auto& pIonServer : ionServers) {
            result.push_back(pIonServer->getSession());
        }

        return result;
    }

    std::vector<std::string> getServerPaths() noexcept override {
        const auto& ionServers = _pContext->getAssetRegistry().getIonServers();

        std::vector<std::string> result;
        result.reserve(ionServers.size());

        for (const auto& pIonServer : ionServers) {
            result.push_back(pIonServer->getPath().GetString());
        }

        return result;
    }

    SetDefaultTokenResult getSetDefaultTokenResult() noexcept override {
        return _pContext->getCesiumIonServerManager().getSetDefaultTokenResult();
    }

    bool isDefaultTokenSet() noexcept override {
        return _pContext->getCesiumIonServerManager().isDefaultTokenSet();
    }

    void createToken(const char* name) noexcept override {
        _pContext->getCesiumIonServerManager().createToken(name);
    }

    void selectToken(const char* id, const char* token) noexcept override {
        CesiumIonClient::Token t{id, "", token};
        _pContext->getCesiumIonServerManager().selectToken(t);
    }

    void specifyToken(const char* token) noexcept override {
        _pContext->getCesiumIonServerManager().specifyToken(token);
    }

    std::optional<AssetTroubleshootingDetails> getAssetTroubleshootingDetails() noexcept override {
        return _pContext->getCesiumIonServerManager().getAssetTroubleshootingDetails();
    }

    std::optional<TokenTroubleshootingDetails> getAssetTokenTroubleshootingDetails() noexcept override {
        return _pContext->getCesiumIonServerManager().getAssetTokenTroubleshootingDetails();
    }

    std::optional<TokenTroubleshootingDetails> getDefaultTokenTroubleshootingDetails() noexcept override {
        return _pContext->getCesiumIonServerManager().getDefaultTokenTroubleshootingDetails();
    }

    void updateTroubleshootingDetails(
        const char* tilesetPath,
        int64_t tilesetIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept override {
        return _pContext->getCesiumIonServerManager().updateTroubleshootingDetails(
            pxr::SdfPath(tilesetPath), tilesetIonAssetId, tokenEventId, assetEventId);
    }

    void updateTroubleshootingDetails(
        const char* tilesetPath,
        int64_t tilesetIonAssetId,
        int64_t imageryIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept override {
        return _pContext->getCesiumIonServerManager().updateTroubleshootingDetails(
            pxr::SdfPath(tilesetPath), tilesetIonAssetId, imageryIonAssetId, tokenEventId, assetEventId);
    }

    std::string printFabricStage() noexcept override {
        return FabricUtil::printFabricStage(_pContext->getFabricStage());
    }

    RenderStatistics getRenderStatistics() noexcept override {
        return _pContext->getRenderStatistics();
    }

    bool creditsAvailable() noexcept override {
        return _pContext->getCreditSystem()->getCreditsToShowThisFrame().size() > 0;
    }

    std::vector<std::pair<std::string, bool>> getCredits() noexcept override {
        const auto& pCreditSystem = _pContext->getCreditSystem();
        const auto& credits = pCreditSystem->getCreditsToShowThisFrame();

        std::vector<std::pair<std::string, bool>> result;
        result.reserve(credits.size());

        for (const auto& item : credits) {
            const auto showOnScreen = pCreditSystem->shouldBeShownOnScreen(item);
            result.emplace_back(pCreditSystem->getHtml(item), showOnScreen);
        }

        return result;
    }

    void creditsStartNextFrame() noexcept override {
        return _pContext->getCreditSystem()->startNextFrame();
    }

    bool isTracingEnabled() noexcept override {
#if CESIUM_TRACING_ENABLED
        return true;
#else
        return false;
#endif
    }

  private:
    std::unique_ptr<Context> _pContext;
};
} // namespace cesium::omniverse

const struct carb::PluginImplDesc pluginImplDesc = {
    "cesium.omniverse.plugin",
    "Cesium Omniverse Carbonite Plugin.",
    "Cesium",
    carb::PluginHotReload::eDisabled,
    "dev"};

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

CARB_PLUGIN_IMPL(pluginImplDesc, cesium::omniverse::CesiumOmniversePlugin)
CARB_PLUGIN_IMPL_DEPS(omni::fabric::IFabric, omni::kit::IApp, carb::settings::ISettings)

#ifdef CESIUM_OMNI_CLANG
#pragma clang diagnostic pop
#endif

void fillInterface([[maybe_unused]] cesium::omniverse::CesiumOmniversePlugin& iface) {}
