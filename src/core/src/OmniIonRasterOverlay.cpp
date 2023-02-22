#include "cesium/omniverse/OmniIonRasterOverlay.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/rasterOverlay.h>

namespace cesium::omniverse {

OmniIonRasterOverlay::OmniIonRasterOverlay(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniIonRasterOverlay::getPath() const {
    return _path;
}

std::string OmniIonRasterOverlay::getName() const {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(_path);
    assert(prim.IsValid());

    return prim.GetName().GetString();
}

int64_t OmniIonRasterOverlay::getIonAssetId() const {
    auto stage = UsdUtil::getUsdStage();
    auto prim = stage->GetPrimAtPath(_path);
    assert(prim.IsValid());

    pxr::CesiumRasterOverlay rasterOverlay(prim);

    int64_t assetId;
    rasterOverlay.GetRasterOverlayIdAttr().Get<int64_t>(&assetId);

    return assetId;
}

std::optional<CesiumIonClient::Token> OmniIonRasterOverlay::getIonToken() const {
    // TODO: Implement actually using the override tokens.
    return Context::instance().getDefaultToken();
}
} // namespace cesium::omniverse
