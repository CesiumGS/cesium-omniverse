#pragma once

#include <pxr/usd/usd/common.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <gsl/span>

namespace omni::fabric {
class StageReaderWriter;
}

namespace CesiumUtility {
class CreditSystem;
}

namespace CesiumAsync {
class AsyncSystem;
class IAssetAccessor;
} // namespace CesiumAsync

namespace cesium::omniverse {

class AssetRegistry;
class CesiumIonServerManager;
class FabricResourceManager;
class Logger;
class TaskProcessor;
class UsdNotificationHandler;
struct RenderStatistics;
struct Viewport;

class Context {
  public:
    Context(const std::filesystem::path& cesiumExtensionLocation);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) noexcept = delete;
    Context& operator=(Context&&) noexcept = delete;

    [[nodiscard]] const std::filesystem::path& getCesiumExtensionLocation() const;
    [[nodiscard]] const std::filesystem::path& getCertificatePath() const;
    [[nodiscard]] const pxr::TfToken& getCesiumMdlPathToken() const;

    [[nodiscard]] std::shared_ptr<TaskProcessor> getTaskProcessor() const;
    [[nodiscard]] const CesiumAsync::AsyncSystem& getAsyncSystem() const;
    [[nodiscard]] std::shared_ptr<CesiumAsync::IAssetAccessor> getHttpAssetAccessor() const;
    [[nodiscard]] std::shared_ptr<CesiumUtility::CreditSystem> getCreditSystem() const;
    [[nodiscard]] std::shared_ptr<Logger> getLogger() const;
    [[nodiscard]] const AssetRegistry& getAssetRegistry() const;
    [[nodiscard]] AssetRegistry& getAssetRegistry();
    [[nodiscard]] const FabricResourceManager& getFabricResourceManager() const;
    [[nodiscard]] FabricResourceManager& getFabricResourceManager();
    [[nodiscard]] const CesiumIonServerManager& getCesiumIonServerManager() const;
    [[nodiscard]] CesiumIonServerManager& getCesiumIonServerManager();

    void clearStage();
    void reloadStage();

    void onUpdateFrame(const gsl::span<const Viewport>& viewports);
    void onUsdStageChanged(int64_t stageId);

    [[nodiscard]] const pxr::UsdStageWeakPtr& getUsdStage() const;
    [[nodiscard]] pxr::UsdStageWeakPtr& getUsdStage();
    [[nodiscard]] int64_t getUsdStageId() const;
    [[nodiscard]] bool hasUsdStage() const;
    [[nodiscard]] omni::fabric::StageReaderWriter& getFabricStage() const;

    [[nodiscard]] RenderStatistics getRenderStatistics() const;

  private:
    std::filesystem::path _cesiumExtensionLocation;
    std::filesystem::path _certificatePath;
    pxr::TfToken _cesiumMdlPathToken;

    std::shared_ptr<TaskProcessor> _pTaskProcessor;
    std::unique_ptr<CesiumAsync::AsyncSystem> _pAsyncSystem;
    std::shared_ptr<CesiumAsync::IAssetAccessor> _pHttpAssetAccessor;
    std::shared_ptr<CesiumUtility::CreditSystem> _pCreditSystem;
    std::shared_ptr<Logger> _pLogger;
    std::unique_ptr<AssetRegistry> _pAssetRegistry;
    std::unique_ptr<FabricResourceManager> _pFabricResourceManager;
    std::unique_ptr<CesiumIonServerManager> _pCesiumIonServerManager;
    std::unique_ptr<UsdNotificationHandler> _pUsdNotificationHandler;

    pxr::UsdStageWeakPtr _pUsdStage;
    std::unique_ptr<omni::fabric::StageReaderWriter> _pFabricStage;
    int64_t _usdStageId{0};
};

} // namespace cesium::omniverse
