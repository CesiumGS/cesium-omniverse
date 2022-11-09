#include "cesium/omniverse/CesiumOmniverse.h"

#include "cesium/omniverse/OmniTileset.h"

#include <unordered_map>

namespace Cesium {

namespace {
int currentId = 0;
std::unordered_map<int, std::unique_ptr<OmniTileset>> tilesets;
} // namespace

void initialize() noexcept {
    OmniTileset::init();
}

void finalize() noexcept {
    OmniTileset::shutdown();
}

int addTilesetUrl(const pxr::UsdStageRefPtr& stage, const char* url) noexcept {
    const int tilesetId = currentId++;
    tilesets.insert({tilesetId, std::make_unique<OmniTileset>(stage, url)});
    return tilesetId;
}

int addTilesetIon(const pxr::UsdStageRefPtr& stage, int64_t ionId, const char* ionToken) noexcept {
    const int tilesetId = currentId++;
    tilesets.insert({tilesetId, std::make_unique<OmniTileset>(stage, ionId, ionToken)});
    return tilesetId;
}

void removeTileset(int tileset) noexcept {
    tilesets.erase(tileset);
}

void updateFrame(
    int tileset,
    const double* viewMatrix,
    const double* projMatrix,
    double width,
    double height) noexcept {
    const auto iter = tilesets.find(tileset);
    if (iter != tilesets.end()) {
        pxr::GfMatrix4d viewMatrixGf(
            viewMatrix[0],
            viewMatrix[1],
            viewMatrix[2],
            viewMatrix[3],
            viewMatrix[4],
            viewMatrix[5],
            viewMatrix[6],
            viewMatrix[7],
            viewMatrix[8],
            viewMatrix[9],
            viewMatrix[10],
            viewMatrix[11],
            viewMatrix[12],
            viewMatrix[13],
            viewMatrix[14],
            viewMatrix[15]);
        pxr::GfMatrix4d projMatrixGf(
            projMatrix[0],
            projMatrix[1],
            projMatrix[2],
            projMatrix[3],
            projMatrix[4],
            projMatrix[5],
            projMatrix[6],
            projMatrix[7],
            projMatrix[8],
            projMatrix[9],
            projMatrix[10],
            projMatrix[11],
            projMatrix[12],
            projMatrix[13],
            projMatrix[14],
            projMatrix[15]);
        iter->second->updateFrame(viewMatrixGf, projMatrixGf, width, height);
    }
}

void setGeoreferenceOrigin(double longitude, double latitude, double height) noexcept {
    Cesium::Georeference::instance().setOrigin(CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(
        CesiumGeospatial::Cartographic(glm::radians(longitude), glm::radians(latitude), height)));
}

} // namespace Cesium
