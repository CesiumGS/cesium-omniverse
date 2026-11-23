#include "cesium/omniverse/Context.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/CertificateAuthorityPaths.h"
#include "cesium/omniverse/CesiumIonServerManager.h"
#include "cesium/omniverse/CesiumIonSessionManager.h"
#include "cesium/omniverse/DebugSettings.h"
#include "cesium/omniverse/FilesystemUtil.h"
#include "cesium/omniverse/ILogger.h"
#include "cesium/omniverse/ISettings.h"
#include "cesium/omniverse/ISubsystem.h"
#include "cesium/omniverse/OmniData.h"
#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/RenderStatistics.h"
#include "cesium/omniverse/SubsystemStatistics.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/TilesetStatistics.h"
#include "cesium/omniverse/UsdNotificationHandler.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <CesiumAsync/CachingAssetAccessor.h>
#include <CesiumAsync/SqliteCache.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumUtility/CreditSystem.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdUtils/stageCache.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

#include <chrono>

namespace cesium::omniverse {

namespace {

std::string getCacheDatabaseName(const std::filesystem::path& cesiumCacheDirectory) {
    if (!cesiumCacheDirectory.empty()) {
        auto cacheFilePath = cesiumCacheDirectory / "cesium-request-cache.sqlite";
        return cacheFilePath.generic_string();
    }
    return {};
}

std::shared_ptr<CesiumAsync::ICacheDatabase> makeCacheDatabase(
    const std::filesystem::path& cesiumCacheDirectory,
    const std::shared_ptr<ILogger>& logger,
    const ISettings& settings) {
    uint64_t maxCacheItems = settings.getMaxCacheItems();
    if (maxCacheItems == 0) {
        logger->oneTimeWarning("maxCacheItems set to 0, so disabling accessor cache");
        return {};
    } else if (auto dbName = getCacheDatabaseName(cesiumCacheDirectory); !dbName.empty()) {
        logger->oneTimeWarning("Cesium cache file: {}", dbName);
        return std::make_shared<CesiumAsync::SqliteCache>(logger, dbName, maxCacheItems);
    }
    logger->oneTimeWarning("could not get name for cache database");
    return {};
}

uint64_t getSecondsSinceEpoch() {
    const auto timePoint = std::chrono::system_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(timePoint.time_since_epoch()).count());
}

CesiumCurl::CurlAssetAccessorOptions createCurlAssetAccessorOptions() {
    CesiumCurl::CurlAssetAccessorOptions options;

    options.certificateFile = getCAFile();
    options.certificatePath = getCAPath();
    options.requestHeaders.emplace_back("X-Cesium-Client", "Cesium for Omniverse");
    options.requestHeaders.emplace_back(
        "X-Cesium-Client-Version", fmt::format("v{} {}", CESIUM_OMNI_VERSION, CESIUM_OMNI_GIT_HASH_ABBREVIATED));

#ifdef OMNI_OMNI_MSVC
    std::string_view platformName = "windows";
#else
    std::string_view platformName = "linux";
#endif

    options.requestHeaders.emplace_back("X-Cesium-Client-OS", platformName);

    return options;
}

} // namespace

Context::Context(std::unique_ptr<ISubsystem>&& pSubsystem, const std::filesystem::path& cesiumCacheDirectory)
    : _pSubsystem(std::move(pSubsystem))
    , _pTaskProcessor(std::make_shared<TaskProcessor>())
    , _pAsyncSystem(std::make_unique<CesiumAsync::AsyncSystem>(_pTaskProcessor))
    , _pCacheDatabase(
          makeCacheDatabase(cesiumCacheDirectory.lexically_normal(), getLogger(), _pSubsystem->getSettings()))
    , _pCreditSystem(std::make_shared<CesiumUtility::CreditSystem>())
    , _pAssetRegistry(std::make_unique<AssetRegistry>(this))
    , _pCesiumIonSessionManager(std::make_unique<CesiumIonSessionManager>(this))
    , _pCesiumIonServerManager(std::make_unique<CesiumIonServerManager>(this))
    , _pUsdNotificationHandler(std::make_unique<UsdNotificationHandler>(this))
    , _contextId(static_cast<int64_t>(getSecondsSinceEpoch())) {
    if (_pCacheDatabase) {
        _pAssetAccessor = std::make_shared<CesiumAsync::CachingAssetAccessor>(
            getLogger(),
            std::make_shared<CesiumCurl::CurlAssetAccessor>(createCurlAssetAccessorOptions()),
            _pCacheDatabase);
    } else {
        _pAssetAccessor = std::make_shared<CesiumCurl::CurlAssetAccessor>(createCurlAssetAccessorOptions());
    }
    Cesium3DTilesContent::registerAllTileContentTypes();

#if CESIUM_TRACING_ENABLED
    const auto timeNow = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::steady_clock::now());
    const auto timeSinceEpoch = timeNow.time_since_epoch().count();
    const auto path = cesiumExtensionLocation / fmt::format("cesium-trace-{}.json", timeSinceEpoch);
    CESIUM_TRACE_INIT(path.string());
#endif
}

Context::~Context() {
    clearStage();

    CESIUM_TRACE_SHUTDOWN();
}

const ISubsystem& Context::getSubsystem() const {
    return *_pSubsystem.get();
}

std::shared_ptr<TaskProcessor> Context::getTaskProcessor() const {
    return _pTaskProcessor;
}

const CesiumAsync::AsyncSystem& Context::getAsyncSystem() const {
    return *_pAsyncSystem.get();
}

std::shared_ptr<CesiumAsync::IAssetAccessor> Context::getAssetAccessor() const {
    return _pAssetAccessor;
}

std::shared_ptr<CesiumUtility::CreditSystem> Context::getCreditSystem() const {
    return _pCreditSystem;
}

std::shared_ptr<ILogger> Context::getLogger() const {
    return _pSubsystem->getLogger();
}

const AssetRegistry& Context::getAssetRegistry() const {
    return *_pAssetRegistry.get();
}

AssetRegistry& Context::getAssetRegistry() {
    return *_pAssetRegistry.get();
}

const CesiumIonSessionManager& Context::getCesiumIonSessionManager() const {
    return *_pCesiumIonSessionManager.get();
}

CesiumIonSessionManager& Context::getCesiumIonSessionManager() {
    return *_pCesiumIonSessionManager.get();
}

const CesiumIonServerManager& Context::getCesiumIonServerManager() const {
    return *_pCesiumIonServerManager.get();
}

CesiumIonServerManager& Context::getCesiumIonServerManager() {
    return *_pCesiumIonServerManager.get();
}

void Context::clearStage() {
    // The order is important. Clear the asset registry first so that subsystem
    // resources are released back into the pool. Then clear the pools.
    _pAssetRegistry->clear();
    _pUsdNotificationHandler->clear();
    _pSubsystem->clear();
}

void Context::reloadStage() {
    clearStage();

    // Populate the asset registry from prims already on the stage
    _pUsdNotificationHandler->onStageLoaded();

    // Apply debug settings
    auto pData = _pAssetRegistry->getFirstData();
    if (pData) {
        _pSubsystem->applyDebugSettings(DebugSettings{
            .disableMaterials = pData->getDebugDisableMaterials(),
            .disableTextures = pData->getDebugDisableTextures(),
            .disableGeometryPool = pData->getDebugDisableGeometryPool(),
            .disableMaterialPool = pData->getDebugDisableMaterialPool(),
            .disableTexturePool = pData->getDebugDisableTexturePool(),
            .geometryPoolInitialCapacity = pData->getDebugGeometryPoolInitialCapacity(),
            .materialPoolInitialCapacity = pData->getDebugMaterialPoolInitialCapacity(),
            .texturePoolInitialCapacity = pData->getDebugTexturePoolInitialCapacity(),
            .randomColors = pData->getDebugRandomColors(),
        });
    }
}

void Context::clearAccessorCache() {
    if (_pCacheDatabase) {
        _pCacheDatabase->clearAll();
    }
}

void Context::onUpdateFrame(const std::span<const Viewport>& viewports, bool waitForLoadingTiles) {
    _pAsyncSystem->dispatchMainThreadTasks();
    _pUsdNotificationHandler->onUpdateFrame();
    _pAssetRegistry->onUpdateFrame(viewports, waitForLoadingTiles);
    _pCesiumIonServerManager->onUpdateFrame();
}

void Context::onUsdStageChanged(int64_t usdStageId) {
    if (_usdStageId == usdStageId) {
        return;
    }

    // Remove references to the old stage
    _pUsdStage = nullptr;
    _pSubsystem->releaseStage();
    _usdStageId = 0;

    // Now it's safe to clear anything else that references the old stage
    clearStage();

    if (usdStageId > 0) {
        _pUsdStage =
            pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(static_cast<long>(usdStageId)));

        _pSubsystem->acquireStage(usdStageId);

        _usdStageId = usdStageId;

        reloadStage();
    }
}

const pxr::UsdStageWeakPtr& Context::getUsdStage() const {
    return _pUsdStage;
}

pxr::UsdStageWeakPtr& Context::getUsdStage() {
    return _pUsdStage;
}

int64_t Context::getUsdStageId() const {
    return _usdStageId;
}

bool Context::hasUsdStage() const {
    return getUsdStageId() != 0;
}

RenderStatistics Context::getRenderStatistics() const {
    RenderStatistics renderStatistics;

    if (!hasUsdStage()) {
        return renderStatistics;
    }

    SubsystemStatistics subsystemStatistics = _pSubsystem->getStatistics();

    renderStatistics.materialsCapacity = subsystemStatistics.materialsCapacity;
    renderStatistics.materialsLoaded = subsystemStatistics.materialsLoaded;
    renderStatistics.geometriesCapacity = subsystemStatistics.geometriesCapacity;
    renderStatistics.geometriesLoaded = subsystemStatistics.geometriesLoaded;
    renderStatistics.geometriesRendered = subsystemStatistics.geometriesRendered;
    renderStatistics.trianglesLoaded = subsystemStatistics.trianglesLoaded;
    renderStatistics.trianglesRendered = subsystemStatistics.trianglesRendered;

    const auto& tilesets = _pAssetRegistry->getTilesets();
    for (const auto& pTileset : tilesets) {
        const auto tilesetStatistics = pTileset->getStatistics();
        renderStatistics.tilesetCachedBytes += tilesetStatistics.tilesetCachedBytes;
        renderStatistics.tilesVisited += tilesetStatistics.tilesVisited;
        renderStatistics.culledTilesVisited += tilesetStatistics.culledTilesVisited;
        renderStatistics.tilesRendered += tilesetStatistics.tilesRendered;
        renderStatistics.tilesCulled += tilesetStatistics.tilesCulled;
        renderStatistics.maxDepthVisited += tilesetStatistics.maxDepthVisited;
        renderStatistics.tilesLoadingWorker += tilesetStatistics.tilesLoadingWorker;
        renderStatistics.tilesLoadingMain += tilesetStatistics.tilesLoadingMain;
        renderStatistics.tilesLoaded += tilesetStatistics.tilesLoaded;
    }

    return renderStatistics;
}

int64_t Context::getContextId() const {
    return _contextId;
}

} // namespace cesium::omniverse
