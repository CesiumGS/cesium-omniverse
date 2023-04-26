#pragma once

#include "cesium/omniverse/FabricStatistics.h"
#include "cesium/omniverse/SetDefaultTokenResult.h"
#include "cesium/omniverse/TokenTroubleshooter.h"

#include <carb/Interface.h>
#include <pxr/pxr.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE
class GfMatrix4d;
PXR_NAMESPACE_CLOSE_SCOPE

namespace cesium::omniverse {

class CesiumIonSession;

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
     * @brief Adds a tileset from url.
     *
     * @param name The user-given name of this tileset.
     * @param url The tileset url.
     * @returns The tileset sdf path.
     */
    virtual std::string addTilesetUrl(const char* name, const char* url) noexcept = 0;

    /**
     * @brief Adds a tileset from ion using the project default ion access token.
     *
     * @param name The user-given name of this tileset.
     * @param ionAssetId The ion asset ID.
     * @returns The tileset sdf path.
     */
    virtual std::string addTilesetIon(const char* name, int64_t ionAssetId) noexcept = 0;

    /**
     * @brief Adds a tileset from ion using the given ion access token.
     *
     * @param name The user-given name of this tileset.
     * @param ionAssetId The ion asset ID.
     * @param ionAccessToken The ion access token.
     * @returns The tileset sdf path.
     */
    virtual std::string addTilesetIon(const char* name, int64_t ionAssetId, const char* ionAccessToken) noexcept = 0;

    /**
     * @brief Adds imagery from ion.
     *
     * @param tilesetPath The sdf path of the tileset that the imagery will be attached to.
     * @param name The user-given name of this imagery.
     * @param ionAssetId The ion asset ID.
     * @returns The imagery sdf path.
     */
    virtual std::string addImageryIon(const char* tilesetPath, const char* name, int64_t ionAssetId) noexcept = 0;

    /**
     * @brief Adds imagery from ion.
     *
     * @param tilesetPath The sdf path of the tileset that the imagery will be attached to.
     * @param name The user-given name of this imagery.
     * @param ionAssetId The ion asset ID.
     * @param ionAccessToken The ion access token.
     * @returns The imagery sdf path.
     */
    virtual std::string addImageryIon(
        const char* tilesetPath,
        const char* name,
        int64_t ionAssetId,
        const char* ionAccessToken) noexcept = 0;

    /**
     * @brief Adds a tileset and imagery from ion.
     *
     * @param tilesetName The user-given name of this tileset.
     * @param tilesetIonAssetId The ion asset ID for the tileset.
     * @param imageryName The user-given name of this imagery.
     * @param imageryIonAssetId The ion asset ID for the imagery.
     * @returns The tileset sdf path.
     */
    virtual std::string addTilesetAndImagery(
        const char* tilesetName,
        int64_t tilesetIonAssetId,
        const char* imageryName,
        int64_t imageryIonAssetId) noexcept = 0;

    /**
     * @brief Gets all the tileset paths on the stage.
     *
     * @return The tileset sdf paths.
     */
    virtual std::vector<std::string> getAllTilesetPaths() noexcept = 0;

    /**
     * @brief Returns true if the given path corresponds to a CesiumTileset prim, otherwise returns false.
     *
     * @param path The sdf path.
     * @return Returns true if the given path corresponds to a CesiumTileset prim, otherwise returns false.
     */
    virtual bool isTileset(const char* path) noexcept = 0;

    /**
     * @brief Removes a tileset from the stage.
     *
     * @param tilesetPath The tileset sdf path. If there's no tileset with this path nothing happens.
     */
    virtual void removeTileset(const char* tilesetPath) noexcept = 0;

    /**
     * @brief Reloads a tileset.
     *
     * @param tilesetPath The tileset sdf path. If there's no tileset with this path nothing happens.
     */
    virtual void reloadTileset(const char* tilesetPath) noexcept = 0;

    /**
     * @brief Updates all tilesets this frame.
     *
     * @param viewMatrix The view matrix.
     * @param projMatrix The projection matrix.
     * @param width The screen width.
     * @param height The screen height.
     */
    virtual void onUpdateFrame(
        const pxr::GfMatrix4d& viewMatrix,
        const pxr::GfMatrix4d& projMatrix,
        double width,
        double height) noexcept = 0;

    /**
     * @brief Updates the UI.
     */
    virtual void onUpdateUi() noexcept = 0;

    /**
     * @brief Updates the reference to the USD stage for the C++ layer.
     *
     * @param stageId The id of the current stage.
     */
    virtual void onStageChange(long stageId) noexcept = 0;

    /**
     * @brief Sets the georeference origin based on the WGS84 ellipsoid.
     *
     * @param longitude The longitude in degrees.
     * @param latitude The latitude in degrees.
     * @param height The height in meters.
     */
    virtual void setGeoreferenceOrigin(double longitude, double latitude, double height) noexcept = 0;

    /**
     * @brief Connects to Cesium ion.
     */
    virtual void connectToIon() noexcept = 0;

    /**
     * @brief Gets the Cesium ion session.
     */
    virtual std::optional<std::shared_ptr<CesiumIonSession>> getSession() noexcept = 0;

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
        int64_t imageryIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept = 0;

    /**
     * @brief Prints the Fabric stage. For debugging only.
     *
     * @returns A string representation of the Fabric stage.
     */
    virtual std::string printFabricStage() noexcept = 0;

    /**
     * @brief Get Fabric statistics. For debugging only.
     *
     * @returns Object containing Fabric statistics.
     */
    virtual FabricStatistics getFabricStatistics() noexcept = 0;

    virtual bool creditsAvailable() noexcept = 0;
    virtual std::vector<std::pair<std::string, bool>> getCredits() noexcept = 0;
};

} // namespace cesium::omniverse
