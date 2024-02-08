#include "cesium/omniverse/Context.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/CesiumIonServerManager.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricStatistics.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/OmniData.h"
#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/RenderStatistics.h"
#include "cesium/omniverse/SettingsWrapper.h"
#include "cesium/omniverse/TaskProcessor.h"
#include "cesium/omniverse/TilesetStatistics.h"
#include "cesium/omniverse/UrlAssetAccessor.h"
#include "cesium/omniverse/UsdNotificationHandler.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesContent/registerAllTileContentTypes.h>
#include <CesiumAsync/CachingAssetAccessor.h>
#include <CesiumAsync/SqliteCache.h>
#include <CesiumUtility/CreditSystem.h>
#include <omni/fabric/SimStageWithHistory.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdUtils/stageCache.h>

#if CESIUM_TRACING_ENABLED
#include <chrono>
#endif

#include <cstdlib>

#if defined(__linux__)
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace cesium::omniverse {

namespace {
// Quite a lot of ceremony to get the home directory
std::string getCacheDatabaseName() {
    std::string homeDir;
#if defined(__linux__)
    if (char* cString = std::getenv("HOME")) {
        homeDir = cString;
    } else {
        passwd pwd;
        long bufsize = sysconf(_SC_GETPW_R_SIZE_MAX);
        if (bufsize == -1) {
            bufsize = 16384;
        }
        char* buf = new char[static_cast<size_t>(bufsize)];
        passwd* result = nullptr;
        int getResult = getpwuid_r(getuid(), &pwd, buf, static_cast<size_t>(bufsize), &result);
        if (getResult == 0) {
            homeDir = pwd.pw_dir;
        }
        delete[] buf;
    }
#elif defined(_WIN32)
    if (char* cString = getenv("USERPROFILE")) {
        homeDir = cString;
    }
#endif
    if (!homeDir.empty()) {
        std::filesystem::path homeDirPath(homeDir);
        auto cacheFilePath = homeDirPath / ".nvidia-omniverse" / "cesium-request-cache.sqlite";
        return cacheFilePath.generic_string();
    }
    return {};
}

std::shared_ptr<CesiumAsync::ICacheDatabase>& getCacheDatabase(const std::shared_ptr<Logger>& logger) {
    // lambda is used here so that real code can be run when the static is initialized.
    static auto pCacheDatabase = [&]() -> std::shared_ptr<CesiumAsync::ICacheDatabase> {
        uint64_t maxCacheItems = Settings::getMaxCacheItems();
        if (maxCacheItems == 0) {
            logger->oneTimeWarning("maxCacheItems set to 0, so disabling accessor cache");
            return {};
        } else if (auto dbName = getCacheDatabaseName(); !dbName.empty()) {
            return std::make_shared<CesiumAsync::SqliteCache>(logger, getCacheDatabaseName(), maxCacheItems);
        }
        logger->oneTimeWarning("could not get name for cache database");
        return {};
    }();
    return pCacheDatabase;
}
} // namespace

Context::Context(const std::filesystem::path& cesiumExtensionLocation)
    : _cesiumExtensionLocation(cesiumExtensionLocation.lexically_normal())
    , _cesiumMdlPathToken(pxr::TfToken((_cesiumExtensionLocation / "mdl" / "cesium.mdl").generic_string()))
    , _pTaskProcessor(std::make_shared<TaskProcessor>())
    , _pAsyncSystem(std::make_unique<CesiumAsync::AsyncSystem>(_pTaskProcessor))
    , _pLogger(std::make_shared<Logger>())
    , _pCacheDatabase(getCacheDatabase(_pLogger))
    , _pCreditSystem(std::make_shared<CesiumUtility::CreditSystem>())
    , _pAssetRegistry(std::make_unique<AssetRegistry>(this))
    , _pFabricResourceManager(std::make_unique<FabricResourceManager>(this))
    , _pCesiumIonServerManager(std::make_unique<CesiumIonServerManager>(this))
    , _pUsdNotificationHandler(std::make_unique<UsdNotificationHandler>(this)) {
    if (_pCacheDatabase) {
        _pAssetAccessor = std::make_shared<CesiumAsync::CachingAssetAccessor>(
            _pLogger, std::make_shared<UrlAssetAccessor>(), _pCacheDatabase);

    } else {
        _pAssetAccessor = std::make_shared<UrlAssetAccessor>();
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

const std::filesystem::path& Context::getCesiumExtensionLocation() const {
    return _cesiumExtensionLocation;
}

const pxr::TfToken& Context::getCesiumMdlPathToken() const {
    return _cesiumMdlPathToken;
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

std::shared_ptr<Logger> Context::getLogger() const {
    return _pLogger;
}

const AssetRegistry& Context::getAssetRegistry() const {
    return *_pAssetRegistry.get();
}

AssetRegistry& Context::getAssetRegistry() {
    return *_pAssetRegistry.get();
}

const FabricResourceManager& Context::getFabricResourceManager() const {
    return *_pFabricResourceManager.get();
}

FabricResourceManager& Context::getFabricResourceManager() {
    return *_pFabricResourceManager.get();
}

const CesiumIonServerManager& Context::getCesiumIonServerManager() const {
    return *_pCesiumIonServerManager.get();
}

CesiumIonServerManager& Context::getCesiumIonServerManager() {
    return *_pCesiumIonServerManager.get();
}

void Context::clearStage() {
    // The order is important. Clear the asset registry first so that Fabric
    // resources are released back into the pool. Then clear the pools.
    _pAssetRegistry->clear();
    _pFabricResourceManager->clear();
    _pUsdNotificationHandler->clear();
}

void Context::reloadStage() {
    const auto defaultIonServerPath = pxr::SdfPath("/CesiumServers/IonOfficial");

    clearStage();

    // Populate the asset registry from prims already on the stage
    _pUsdNotificationHandler->onStageLoaded();

    // Update FabricResourceManager settings
    auto pData = _pAssetRegistry->getFirstData();
    if (pData) {
        _pFabricResourceManager->setDisableMaterials(pData->getDebugDisableMaterials());
        _pFabricResourceManager->setDisableTextures(pData->getDebugDisableTextures());
        _pFabricResourceManager->setDisableGeometryPool(pData->getDebugDisableGeometryPool());
        _pFabricResourceManager->setDisableMaterialPool(pData->getDebugDisableMaterialPool());
        _pFabricResourceManager->setDisableTexturePool(pData->getDebugDisableTexturePool());
        _pFabricResourceManager->setGeometryPoolInitialCapacity(pData->getDebugGeometryPoolInitialCapacity());
        _pFabricResourceManager->setMaterialPoolInitialCapacity(pData->getDebugMaterialPoolInitialCapacity());
        _pFabricResourceManager->setTexturePoolInitialCapacity(pData->getDebugTexturePoolInitialCapacity());
        _pFabricResourceManager->setDebugRandomColors(pData->getDebugRandomColors());
    }
}

void Context::clearAccessorCache() {
    if (_pCacheDatabase) {
        _pCacheDatabase->clearAll();
    }
}

void Context::onUpdateFrame(const gsl::span<const Viewport>& viewports) {
    _pUsdNotificationHandler->onUpdateFrame();
    _pAssetRegistry->onUpdateFrame(viewports);
    _pCesiumIonServerManager->onUpdateFrame();
}

void Context::onUsdStageChanged(int64_t usdStageId) {
    if (_usdStageId == usdStageId) {
        return;
    }

    // Remove references to the old stage
    _pUsdStage = nullptr;
    _pFabricStage = nullptr;
    _usdStageId = 0;

    // Now it's safe to clear anything else that references the old stage
    clearStage();

    if (usdStageId > 0) {
        _pUsdStage =
            pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(static_cast<long>(usdStageId)));

        const auto iFabricStage = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
        const auto fabricStageId = iFabricStage->get(omni::fabric::UsdStageId(static_cast<uint64_t>(usdStageId)));
        _pFabricStage = std::make_unique<omni::fabric::StageReaderWriter>(fabricStageId);

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

omni::fabric::StageReaderWriter& Context::getFabricStage() const {
    return *_pFabricStage.get();
}

RenderStatistics Context::getRenderStatistics() const {
    RenderStatistics renderStatistics;

    if (!hasUsdStage()) {
        return renderStatistics;
    }

    const auto fabricStatistics = FabricUtil::getStatistics(getFabricStage());
    renderStatistics.materialsCapacity = fabricStatistics.materialsCapacity;
    renderStatistics.materialsLoaded = fabricStatistics.materialsLoaded;
    renderStatistics.geometriesCapacity = fabricStatistics.geometriesCapacity;
    renderStatistics.geometriesLoaded = fabricStatistics.geometriesLoaded;
    renderStatistics.geometriesRendered = fabricStatistics.geometriesRendered;
    renderStatistics.trianglesLoaded = fabricStatistics.trianglesLoaded;
    renderStatistics.trianglesRendered = fabricStatistics.trianglesRendered;

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

} // namespace cesium::omniverse
