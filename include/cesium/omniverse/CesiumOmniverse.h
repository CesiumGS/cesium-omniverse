#pragma once

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
     * @brief Call this before any tilesets are created.
     *
     * @param cesiumMemLocation The folder containing mem.cesium
     */
    virtual void onStartup(const char* cesiumMemLocation) noexcept = 0;

    /**
     * @brief Call this to free resources on program exist.
     */
    virtual void onShutdown() noexcept = 0;

    /**
     * @brief Adds a Cesium data prim if it does not exist. Always sets the data prim to the specified token.
     *
     * @param token The project default token.
     */
    virtual void addCesiumDataIfNotExists(const char* token) noexcept = 0;

    /**
     * @brief Adds a tileset from url.
     *
     * @param url The tileset url
     * @returns The tileset id. Returns -1 on error.
     */
    virtual int64_t addTilesetUrl(const char* url) noexcept = 0;

    /**
     * @brief Adds a tileset from ion using the stage default token.
     *
     * @param name The user-given name of this tileset.
     * @param ionId The ion asset ID for the tileset.
     * @returns The tileset id. Returns -1 on error.
     */
    virtual int64_t addTilesetIon(const char* name, int64_t ionId) noexcept = 0;

    /**
     * @brief Adds a tileset from ion using the Stage level ion token.
     *
     * @param name The user-given name of this tileset.
     * @param ionId The ion asset id
     * @param ionToken The access token
     * @returns The tileset id. Returns -1 on error.
     */
    virtual int64_t addTilesetIon(const char* name, int64_t ionId, const char* ionToken) noexcept = 0;

    /**
     * @brief Adds a raster overlay from ion.
     *
     * @param tilesetId The tileset id
     * @param name The user-given name of this overlay layer
     * @param ionId The asset ID
     */
    virtual void addIonRasterOverlay(int64_t tilesetId, const char* name, int64_t ionId) noexcept = 0;

    /**
     * @brief Adds a raster overlay from ion.
     *
     * @param tileset The tileset id
     * @param name The user-given name of this overlay layer
     * @param ionId The asset ID
     * @param ionToken The access token
     */
    virtual void
    addIonRasterOverlay(int64_t tilesetId, const char* name, int64_t ionId, const char* ionToken) noexcept = 0;

    /**
     * @brief Adds a tileset and a raster overlay to the stage.
     *
     * @param tilesetName The user-given name of this tileset.
     * @param tilesetIonId The ion asset ID for the tileset.
     * @param rasterOverlayName The user-given name of this overlay layer.
     * @param rasterOverlayIonId The ion asset ID for the raster overlay.
     * @returns The tileset id. Returns -1 on error.
     */
    virtual int64_t addTilesetAndRasterOverlay(
        const char* tilesetName,
        int64_t tilesetIonId,
        const char* rasterOverlayName,
        int64_t rasterOverlayIonId) noexcept = 0;

    /**
     * @brief Gets all the tileset ids and their paths. Primarily for usage on the python end.
     *
     * @return The tileset IDs and their sdf paths, as a vector of pairs.
     */
    virtual std::vector<std::pair<int64_t, const char*>> getAllTilesetIdsAndPaths() noexcept = 0;

    /**
     * @brief Removes a tileset from the scene.
     *
     * @param tilesetId The tileset id. If there's no tileset with this id nothing happens.
     */
    virtual void removeTileset(int64_t tilesetId) noexcept = 0;

    /**
     * @brief Reloads a tileset.
     *
     * @param tilesetId The tileset id
     */
    virtual void reloadTileset(int64_t tilesetId) noexcept = 0;

    /**
     * @brief Updates all tilesets this frame.
     *
     * @param viewMatrix The view matrix
     * @param projMatrix The projection matrix
     * @param width The screen width
     * @param height The screen height
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
     * @param longitude The longitude in degrees
     * @param latitude The latitude in degrees
     * @param height The height in meters
     */
    virtual void setGeoreferenceOrigin(double longitude, double latitude, double height) noexcept = 0;

    virtual void connectToIon() noexcept = 0;

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
     * @return True if default token is set.
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

    virtual void
    updateTroubleshootingDetails(int64_t tilesetId, uint64_t tokenEventId, uint64_t assetEventId) noexcept = 0;

    virtual void updateTroubleshootingDetails(
        int64_t tilesetId,
        int64_t rasterOverlayId,
        uint64_t tokenEventId,
        uint64_t assetEventId) noexcept = 0;

    /**
     * @brief For debugging only. Print the Fabric stage.
     *
     * @returns A string representation of the Fabric stage.
     */
    virtual std::string printFabricStage() noexcept = 0;
};

} // namespace cesium::omniverse
