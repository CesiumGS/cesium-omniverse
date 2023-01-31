#pragma once

#include "cesium/omniverse/CesiumIonSession.h"

#include <CesiumIonClient/Connection.h>
#include <carb/Interface.h>
#include <pxr/pxr.h>

#include <cstdint>
#include <optional>

PXR_NAMESPACE_OPEN_SCOPE
class GfMatrix4d;
PXR_NAMESPACE_CLOSE_SCOPE

namespace cesium::omniverse {

class ICesiumOmniverseInterface {
  public:
    CARB_PLUGIN_INTERFACE("cesium::omniverse::ICesiumOmniverseInterface", 0, 0);

    /**
     * @brief Call this before any tilesets are created.
     *
     * @param cesiumMemLocation The folder containing mem.cesium
     */
    virtual void initialize(const char* cesiumMemLocation) noexcept = 0;

    /**
     * @brief Call this to free resources on program exist.
     */
    virtual void finalize() noexcept = 0;

    /**
     * @brief Adds a Cesium Data prim.
     *
     * @param stageId The USD stage id
     * @param ionToken The Stage level ionToken
     */
    virtual void addCesiumData(long stageId, const char* ionToken) noexcept = 0;

    /**
     * @brief Adds a tileset from url.
     *
     * @param stageId The USD stage id
     * @param url The tileset url
     * @returns The tileset id. Returns -1 on error.
     */
    virtual int addTilesetUrl(long stageId, const char* url) noexcept = 0;

    /**
     * @brief Adds a tileset from ion using the Stage level ion token.
     *
     * @param stageId The USD stage id
     * @param ionId The ion asset id
     * @param ionToken The access token
     * @returns The tileset id. Returns -1 on error.
     */
    virtual int addTilesetIon(long stageId, int64_t ionId, const char* ionToken) noexcept = 0;

    /**
     * @brief Removes a tileset from the scene.
     *
     * @param tileset The tileset id. If there's no tileset with this id nothing happens.
     */
    virtual void removeTileset(int tileset) noexcept = 0;

    /**
     * @brief Adds a raster overlay from ion.
     *
     * @param tileset The tileset id
     * @param name The user-given name of this overlay layer
     * @param ionId The asset ID
     * @param ionToken The access token
     */
    virtual void addIonRasterOverlay(int tileset, const char* name, int64_t ionId, const char* ionToken) noexcept = 0;

    /**
     * @brief Updates the tileset this frame.
     *
     * @param tileset The tileset id. If there's no tileset with this id nothing happens.
     * @param viewMatrix The view matrix.
     * @param projMatrix The projection matrix.
     * @param width The screen width
     * @param height The screen height
     */
    virtual void updateFrame(
        int tileset,
        const pxr::GfMatrix4d& viewMatrix,
        const pxr::GfMatrix4d& projMatrix,
        double width,
        double height) noexcept = 0;

    /**
     * @brief Sets the georeference origin based on the WGS84 ellipsoid.
     *
     * @param longitude The longitude in degrees
     * @param latitude The latitude in degrees
     * @param height The height in meters
     */
    virtual void setGeoreferenceOrigin(double longitude, double latitude, double height) noexcept = 0;

    virtual void connectToIon() noexcept = 0;

    virtual void onUiUpdate() noexcept = 0;

    virtual std::optional<std::shared_ptr<CesiumIonSession>> getSession() noexcept = 0;
};

} // namespace cesium::omniverse
