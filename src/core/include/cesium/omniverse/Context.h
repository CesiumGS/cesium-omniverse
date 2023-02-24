#pragma once

#include "cesium/omniverse/SetDefaultTokenResult.h"
#include "cesium/omniverse/TokenTroubleshooter.h"

#include <CesiumGeospatial/Cartographic.h>
#include <carb/flatcache/StageWithHistory.h>
#include <glm/glm.hpp>
#include <pxr/usd/usd/common.h>
#include <spdlog/logger.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <vector>

namespace Cesium3DTilesSelection {
class CreditSystem;
class ViewState;
} // namespace Cesium3DTilesSelection

namespace CesiumGeospatial {
class Cartographic;
}

namespace CesiumIonClient {
struct Token;
}

namespace cesium::omniverse {

class CesiumIonSession;
class HttpAssetAccessor;
class OmniTileset;
class TaskProcessor;

class Context {
  public:
    static void onStartup(const std::filesystem::path& cesiumExtensionLocation);
    static void onShutdown();
    static Context& instance();

    Context() = default;
    ~Context() = default;
    Context(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(const Context&) = delete;
    Context& operator=(Context&&) = delete;

    void initialize(int64_t contextId, const std::filesystem::path& cesiumExtensionLocation);
    void destroy();

    std::shared_ptr<TaskProcessor> getTaskProcessor();
    std::shared_ptr<HttpAssetAccessor> getHttpAssetAccessor();
    std::shared_ptr<Cesium3DTilesSelection::CreditSystem> getCreditSystem();
    std::shared_ptr<spdlog::logger> getLogger();

    void addCesiumDataIfNotExists(const CesiumIonClient::Token& token);
    int64_t addTilesetUrl(const std::string& url);
    int64_t addTilesetIon(const std::string& name, int64_t ionId, const std::string& ionToken);
    void addIonRasterOverlay(int64_t tilesetId, const std::string& name, int64_t ionId, const std::string& ionToken);

    void removeTileset(int64_t tilesetId);
    void reloadTileset(int64_t tilesetId);

    void onUpdateFrame(const glm::dmat4& viewMatrix, const glm::dmat4& projMatrix, double width, double height);
    void onUpdateUi();

    pxr::UsdStageRefPtr getStage() const;
    carb::flatcache::StageInProgress getFabricStageInProgress() const;
    long getStageId() const;

    void setStageId(long stageId);

    int64_t getContextId() const;

    const CesiumGeospatial::Cartographic& getGeoreferenceOrigin() const;
    void setGeoreferenceOrigin(const CesiumGeospatial::Cartographic& origin);

    void connectToIon();
    std::optional<std::shared_ptr<CesiumIonSession>> getSession();

    std::optional<CesiumIonClient::Token> getDefaultToken() const;
    SetDefaultTokenResult getSetDefaultTokenResult() const;
    bool isDefaultTokenSet() const;
    void createToken(const std::string& name);
    void selectToken(const CesiumIonClient::Token& token);
    void specifyToken(const std::string& token);

    std::optional<AssetTroubleshootingDetails> getAssetTroubleshootingDetails();
    std::optional<TokenTroubleshootingDetails> getAssetTokenTroubleshootingDetails();
    std::optional<TokenTroubleshootingDetails> getDefaultTokenTroubleshootingDetails();
    void updateTroubleshootingDetails(int64_t tilesetId, uint64_t tokenEventId, uint64_t assetEventId);
    void updateTroubleshootingDetails(
        int64_t tilesetId,
        int64_t rasterOverlayId,
        uint64_t tokenEventId,
        uint64_t assetEventId);

    std::filesystem::path getCesiumExtensionLocation() const;
    std::filesystem::path getMemCesiumPath() const;
    std::filesystem::path getCertificatePath() const;
    bool getDebugDisableMaterials() const;

  private:
    std::shared_ptr<TaskProcessor> _taskProcessor;
    std::shared_ptr<HttpAssetAccessor> _httpAssetAccessor;
    std::shared_ptr<Cesium3DTilesSelection::CreditSystem> _creditSystem;
    std::shared_ptr<spdlog::logger> _logger;

    std::shared_ptr<CesiumIonSession> _session;
    SetDefaultTokenResult _lastSetTokenResult;

    std::optional<AssetTroubleshootingDetails> _assetTroubleshootingDetails = std::nullopt;
    std::optional<TokenTroubleshootingDetails> _assetTokenTroubleshootingDetails = std::nullopt;
    std::optional<TokenTroubleshootingDetails> _defaultTokenTroubleshootingDetails = std::nullopt;

    pxr::UsdStageRefPtr _stage;
    std::optional<carb::flatcache::StageInProgress> _fabricStageInProgress;
    long _stageId{0};

    int64_t _contextId;

    std::atomic<int64_t> _tilesetId{};

    std::filesystem::path _cesiumExtensionLocation;
    std::filesystem::path _memCesiumPath;
    std::filesystem::path _certificatePath;

    CesiumGeospatial::Cartographic _georeferenceOrigin{0.0, 0.0, 0.0};

    bool _debugDisableMaterials{false};

    std::vector<Cesium3DTilesSelection::ViewState> _viewStates;
};

} // namespace cesium::omniverse
