#pragma once

#include "CesiumOmniverseAbi.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/usd/usd/stage.h>

#include <cstdint>
#include <string>

namespace Cesium {

/**
 * @brief Gets a number. For testing purposes.
 */
CESIUM_OMNI_EXPORT_C_FUNCTION(int)
getNumber() CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Call this before any tilesets are created.
 */
CESIUM_OMNI_EXPORT_C_FUNCTION(void)
initialize() CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Call this to free resources on program exist.
 */
CESIUM_OMNI_EXPORT_C_FUNCTION(void)
finalize() CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Adds a tileset from url.
 *
 * @param stage The USD stage
 * @param url The tileset url
 * @returns The tileset id. Returns -1 on error.
 */
CESIUM_OMNI_EXPORT_CPP_FUNCTION(int)
addTilesetUrl(const pxr::UsdStageRefPtr& stage, const std::string& url) CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Adds a tileset from ion.
 *
 * @param stage The USD stage
 * @param ionId The ion asset id
 * @param ionToken The ion access token
 * @returns The tileset id. Returns -1 on error.
 */
CESIUM_OMNI_EXPORT_CPP_FUNCTION(int)
addTilesetIon(const pxr::UsdStageRefPtr& stage, int64_t ionId, const std::string& ionToken) CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Removes a tileset from the scene.
 *
 * @param tileset The tileset id. If there's no tileset with this id nothing happens.
 */
CESIUM_OMNI_EXPORT_C_FUNCTION(void)
removeTileset(int tileset) CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Updates the tileset this frame.
 *
 * @param tileset The tileset id. If there's no tileset with this id nothing happens.
 * @param viewMatrix The view matrix
 * @param projMatrix The projection matrix
 * @param width The screen width
 * @param height The screen height
 */
CESIUM_OMNI_EXPORT_CPP_FUNCTION(void)
updateFrame(
    int tileset,
    const pxr::GfMatrix4d& viewMatrix,
    const pxr::GfMatrix4d& projMatrix,
    double width,
    double height) CESIUM_OMNI_NOEXCEPT;

/**
 * @brief Sets the georeference origin based on the WGS84 ellipsoid.
 *
 * @param longitude The longitude in degrees
 * @param latitude The latitude in degrees
 * @param height The height in meters
 */
CESIUM_OMNI_EXPORT_C_FUNCTION(void)
setGeoreferenceOrigin(double longitude, double latitude, double height) CESIUM_OMNI_NOEXCEPT;

} // namespace Cesium
