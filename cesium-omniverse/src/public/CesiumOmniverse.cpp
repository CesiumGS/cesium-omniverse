#include "cesium/omniverse/CesiumOmniverse.h"

int getNumber() noexcept {
    return 2;
}

void startup() noexcept {}

void shutdown() noexcept {}

int addTilesetUrl(const pxr::UsdStageRefPtr& stage, const std::string& url) noexcept {
    return 0;
}

int addTilesetIon(const pxr::UsdStageRefPtr& stage, int64_t ionId, const std::string& ionToken) noexcept {
    return 0;
}

void removeTileset(int tileset) noexcept {}

void updateFrame(
    int tileset,
    const pxr::GfMatrix4d& viewMatrix,
    const pxr::GfMatrix4d& projMatrix,
    double width,
    double height) noexcept {}

void setGeoreferenceOrigin(const pxr::GfVec3d& origin) noexcept {}
