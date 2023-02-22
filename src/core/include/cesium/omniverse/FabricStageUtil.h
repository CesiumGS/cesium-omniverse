#pragma once

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

#include <vector>

namespace CesiumGltf {
struct ImageCesium;
struct Model;
} // namespace CesiumGltf

namespace CesiumGeometry {
struct Rectangle;
}

namespace cesium::omniverse::FabricStageUtil {

struct AddTileResults {
    std::vector<pxr::SdfPath> geomPaths;
    std::vector<pxr::SdfPath> allPrimPaths;
    std::vector<std::string> textureAssetNames;
};

AddTileResults addTile(
    int64_t tilesetId,
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& tileTransform,
    const CesiumGltf::Model& model);

AddTileResults addTileWithRasterOverlay(
    int64_t tilesetId,
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& tileTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::ImageCesium& rasterOverlayImage,
    const std::string& rasterOverlayName,
    const CesiumGeometry::Rectangle& rasterOverlayRectangle,
    const glm::dvec2& rasterOverlayUvTranslation,
    const glm::dvec2& rasterOverlayUvScale,
    uint64_t rasterOverlayUvSetIndex);

void removeTile(const std::vector<pxr::SdfPath>& allPrimPaths, const std::vector<std::string>& textureAssetNames);
void setTileVisibility(const std::vector<pxr::SdfPath>& geomPaths, bool visible);
void removeTileset(int64_t tilesetId);
void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform);
}; // namespace cesium::omniverse::FabricStageUtil
