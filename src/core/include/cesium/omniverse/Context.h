#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/usd/common.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

#include <filesystem>
#include <memory>
#include <span>

namespace CesiumUtility {
class CreditSystem;
}

namespace CesiumAsync {
class AsyncSystem;
class IAssetAccessor;
class ICacheDatabase;
} // namespace CesiumAsync

namespace cesium::omniverse {

class AssetRegistry;
class CesiumIonServerManager;
class CesiumIonSessionManager;
class ISubsystem;
class ILogger;
class TaskProcessor;
class UsdNotificationHandler;
struct RenderStatistics;
struct Viewport;

class Context {
  public:
    Context(std::unique_ptr<ISubsystem>&& pSubsystem, const std::filesystem::path& cesiumCacheDirectory);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept = delete;
    Context& operator=(Context&&) noexcept = delete;

    [[nodiscard]] const ISubsystem& getSubsystem() const;
    [[nodiscard]] ISubsystem& getSubsystem();
    [[nodiscard]] std::shared_ptr<TaskProcessor> getTaskProcessor() const;
    [[nodiscard]] const CesiumAsync::AsyncSystem& getAsyncSystem() const;
    [[nodiscard]] std::shared_ptr<CesiumAsync::IAssetAccessor> getAssetAccessor() const;
    [[nodiscard]] std::shared_ptr<CesiumUtility::CreditSystem> getCreditSystem() const;
    [[nodiscard]] std::shared_ptr<ILogger> getLogger() const;
    [[nodiscard]] const AssetRegistry& getAssetRegistry() const;
    [[nodiscard]] AssetRegistry& getAssetRegistry();
    [[nodiscard]] const CesiumIonSessionManager& getCesiumIonSessionManager() const;
    [[nodiscard]] CesiumIonSessionManager& getCesiumIonSessionManager();
    [[nodiscard]] const CesiumIonServerManager& getCesiumIonServerManager() const;
    [[nodiscard]] CesiumIonServerManager& getCesiumIonServerManager();

    void clearStage();
    void reloadStage();
    void clearAccessorCache();

    void onUpdateFrame(const std::span<const Viewport>& viewports, bool waitForLoadingTiles);
    void onUsdStageChanged(int64_t stageId);

    [[nodiscard]] const pxr::UsdStageWeakPtr& getUsdStage() const;
    [[nodiscard]] pxr::UsdStageWeakPtr& getUsdStage();
    [[nodiscard]] int64_t getUsdStageId() const;
    [[nodiscard]] bool hasUsdStage() const;

    [[nodiscard]] RenderStatistics getRenderStatistics() const;

    [[nodiscard]] int64_t getContextId() const;

  private:
    std::shared_ptr<ISubsystem> _pSubsystem;
    std::shared_ptr<TaskProcessor> _pTaskProcessor;
    std::unique_ptr<CesiumAsync::AsyncSystem> _pAsyncSystem;
    std::shared_ptr<CesiumAsync::IAssetAccessor> _pAssetAccessor;
    std::shared_ptr<CesiumAsync::ICacheDatabase> _pCacheDatabase;
    std::shared_ptr<CesiumUtility::CreditSystem> _pCreditSystem;
    std::unique_ptr<AssetRegistry> _pAssetRegistry;
    std::unique_ptr<CesiumIonSessionManager> _pCesiumIonSessionManager;
    std::unique_ptr<CesiumIonServerManager> _pCesiumIonServerManager;
    std::unique_ptr<UsdNotificationHandler> _pUsdNotificationHandler;

    int64_t _contextId;

    pxr::UsdStageWeakPtr _pUsdStage;
    int64_t _usdStageId{0};
};

} // namespace cesium::omniverse
