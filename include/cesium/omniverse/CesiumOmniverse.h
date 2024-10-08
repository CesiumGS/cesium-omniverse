#pragma once

#include "cesium/omniverse/AssetTroubleshootingDetails.h"
#include "cesium/omniverse/RenderStatistics.h"
#include "cesium/omniverse/SetDefaultTokenResult.h"
#include "cesium/omniverse/TokenTroubleshootingDetails.h"

#include <carb/Interface.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace cesium::omniverse {

class CesiumIonSession;

struct ViewportApi {
    double viewMatrix[16]; // NOLINT(modernize-avoid-c-arrays)
    double projMatrix[16]; // NOLINT(modernize-avoid-c-arrays)
    double width;
    double height;
};

class ICesiumOmniverseInterface {
  public:
    CARB_PLUGIN_INTERFACE("cesium::omniverse::ICesiumOmniverseInterface", 0, 0);

    /**
     * @brief Call this on extension startup.
     *
     * @param cesiumExtensionLocation Path to the Cesium Omniverse extension location.
     */
    virtual void onStartup(const char* cesiumExtensionLocation) noexcept = 0;

    /**
     * @brief Call this on extension shutdown.
     */
    virtual void onShutdown() noexcept = 0;

    /**
     * @brief Reloads a tileset.
     *
     * @param tilesetPath The tileset sdf path. If there's no tileset with this path nothing happens.
     */
    virtual void reloadTileset(const char* tilesetPath) noexcept = 0;

    /**
     * @brief Updates all tilesets this frame.
     *
     * @param viewports The viewports.
     * @param count The number of viewports.
     * @param waitForLoadingTiles Whether to wait for all tiles to load before continuing.
     */
    virtual void onUpdateFrame(const ViewportApi* viewports, uint64_t count, bool waitForLoadingTiles) noexcept = 0;

    /**
     * @brief Updates the reference to the USD stage for the C++ layer.
     *
     * @param stageId The id of the current stage.
     */
    virtual void onUsdStageChanged(long stageId) noexcept = 0;

    /**
     * @brief Connects to Cesium ion.
     */
    virtual void connectToIon() noexcept = 0;

    /**
     * @brief Gets the active Cesium ion session.
     */
    virtual std::optional<std::shared_ptr<CesiumIonSession>> getSession() noexcept = 0;

    /**
     * @brief Get the path of the active Cesium ion server.
     */
    virtual std::string getServerPath() noexcept = 0;

    /**
     * @brief Gets all Cesium ion sessions.
     */
    virtual std::vector<std::shared_ptr<CesiumIonSession>> getSessions() noexcept = 0;

    /**
     * @brief Get all Cesium ion server paths.
     */
    virtual std::vector<std::string> getServerPaths() noexcept = 0;

    /**
     * @brief Gets the last result with code and message of setting the default token.
     *
     * @return A struct with a code and message. 0 is successful.
     */
    virtual SetDefaultTokenResult getSetDefaultTokenResult() noexcept = 0;

    /**
     * @brief Boolean to check if the default token is set.
     *
     * @return True if the default token is set.
     */
    virtual bool isDefaultTokenSet() noexcept = 0;

    /**
     * @brief Creates a new token using the specified name.
     *
     * @param name The name for the new token.
     */
    virtual void createToken(const char* name) noexcept = 0;

    /**
     * @brief Selects an existing token associated with the logged in account.
     *
     * @param id The ID of the selected token.
     */
    virtual void selectToken(const char* id, const char* token) noexcept = 0;

    /**
     * @brief Used for the specify token action by the set project default token window.
     *
     * @param token The desired token.
     */
    virtual void specifyToken(const char* token) noexcept = 0;

    virtual std::optional<AssetTroubleshootingDetails> getAssetTroubleshootingDetails() noexcept = 0;

    virtual std::optional<TokenTroubleshootingDetails> getAssetTokenTroubleshootingDetails() noexcept = 0;

    virtual std::optional<TokenTroubleshootingDetails> getDefaultTokenTroubleshootingDetails() noexcept = 0;

    virtual void updateTroubleshootingDetails(
        const char* tilesetPath,
        int64_t tilesetIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept = 0;

    virtual void updateTroubleshootingDetails(
        const char* tilesetPath,
        int64_t tilesetIonAssetId,
        int64_t rasterOverlayIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept = 0;

    /**
     * @brief Prints the Fabric stage. For debugging only.
     *
     * @returns A string representation of the Fabric stage.
     */
    virtual std::string printFabricStage() noexcept = 0;

    /**
     * @brief Get render statistics. For debugging only.
     *
     * @returns Object containing render statistics.
     */
    virtual RenderStatistics getRenderStatistics() noexcept = 0;

    virtual bool creditsAvailable() noexcept = 0;
    virtual std::vector<std::pair<std::string, bool>> getCredits() noexcept = 0;
    virtual void creditsStartNextFrame() noexcept = 0;

    virtual bool isTracingEnabled() noexcept = 0;
    /**
     * @brief Clear the asset accessor cache.
     */
    virtual void clearAccessorCache() = 0;
};

} // namespace cesium::omniverse
