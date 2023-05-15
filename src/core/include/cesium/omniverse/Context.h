#pragma once

#include "cesium/omniverse/SetDefaultTokenResult.h"
#include "cesium/omniverse/TokenTroubleshooter.h"
#include "cesium/omniverse/UsdNotificationHandler.h"

#include <CesiumGeospatial/Cartographic.h>
#include <glm/glm.hpp>
#include <omni/fabric/SimStageWithHistory.h>
#include <pxr/usd/usd/common.h>
#include <spdlog/logger.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <vector>

namespace Cesium3DTilesSelection {
class CreditSystem;
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
struct Viewport;

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

    void setProjectDefaultToken(const CesiumIonClient::Token& token);
    pxr::SdfPath addTilesetUrl(const std::string& name, const std::string& url);
    pxr::SdfPath addTilesetIon(const std::string& name, int64_t ionAssetId, const std::string& ionAccessToken);
    pxr::SdfPath addImageryIon(
        const pxr::SdfPath& tilesetPath,
        const std::string& name,
        int64_t ionAssetId,
        const std::string& ionAccessToken);

    void removeTileset(const pxr::SdfPath& tilesetPath);
    void reloadTileset(const pxr::SdfPath& tilesetPath);
    void clearStage();
    void reloadStage();

    void onUpdateFrame(const std::vector<Viewport>& viewports);
    void onUpdateUi();

    pxr::UsdStageRefPtr getStage() const;
    omni::fabric::StageReaderWriter getFabricStageReaderWriter() const;
    long getStageId() const;

    void setStageId(long stageId);

    int64_t getContextId() const;
    int64_t getNextTilesetId() const;
    int64_t getNextTileId() const;

    const CesiumGeospatial::Cartographic getGeoreferenceOrigin() const;
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
    void updateTroubleshootingDetails(
        const pxr::SdfPath& tilesetPath,
        int64_t tilesetIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId);
    void updateTroubleshootingDetails(
        const pxr::SdfPath& tilesetPath,
        [[maybe_unused]] int64_t tilesetIonAssetId,
        int64_t imageryIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId);

    std::filesystem::path getCesiumExtensionLocation() const;
    std::filesystem::path getCertificatePath() const;
    bool getDebugDisableMaterials() const;

    bool creditsAvailable() const;
    std::vector<std::pair<std::string, bool>> getCredits() const;
    void creditsStartNextFrame();

  private:
    void processPropertyChanged(const ChangedPrim& changedPrim);
    void processPrimRemoved(const ChangedPrim& changedPrim);
    void processPrimAdded(const ChangedPrim& changedPrim);
    void processUsdNotifications();

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
    std::optional<omni::fabric::StageReaderWriter> _fabricStageReaderWriter;
    long _stageId{0};
    UsdNotificationHandler _usdNotificationHandler;

    int64_t _contextId;

    mutable std::atomic<int64_t> _tilesetId{};
    mutable std::atomic<int64_t> _tileId{};

    std::filesystem::path _cesiumExtensionLocation;
    std::filesystem::path _certificatePath;

    bool _debugDisableMaterials{false};
};

} // namespace cesium::omniverse
