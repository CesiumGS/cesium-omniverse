#include "cesium/omniverse/OmniData.h"

#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/data.h>

namespace cesium::omniverse {

OmniData::OmniData(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniData::getPath() const {
    return _path;
}

pxr::SdfPath OmniData::getSelectedIonServer() const {
    const auto data = UsdUtil::getCesiumData(_path);

    pxr::SdfPathVector targets;
    data.GetSelectedIonServerRel().GetForwardedTargets(&targets);

    if (targets.size() < 1) {
        return {};
    }

    return targets[0];
}

bool OmniData::getDebugDisableMaterials() const {
    const auto data = UsdUtil::getCesiumData(_path);

    bool disableMaterials;
    data.GetDebugDisableMaterialsAttr().Get(&disableMaterials);

    return disableMaterials;
}

bool OmniData::getDebugDisableTextures() const {
    const auto data = UsdUtil::getCesiumData(_path);

    bool disableTextures;
    data.GetDebugDisableTexturesAttr().Get(&disableTextures);

    return disableTextures;
}

bool OmniData::getDebugDisableGeometryPool() const {
    const auto data = UsdUtil::getCesiumData(_path);

    bool disableGeometryPool;
    data.GetDebugDisableGeometryPoolAttr().Get(&disableGeometryPool);

    return disableGeometryPool;
}

bool OmniData::getDebugDisableMaterialPool() const {
    const auto data = UsdUtil::getCesiumData(_path);

    bool disableMaterialPool;
    data.GetDebugDisableMaterialPoolAttr().Get(&disableMaterialPool);

    return disableMaterialPool;
}

bool OmniData::getDebugDisableTexturePool() const {
    const auto data = UsdUtil::getCesiumData(_path);

    bool disableTexturePool;
    data.GetDebugDisableTexturePoolAttr().Get(&disableTexturePool);

    return disableTexturePool;
}

uint64_t OmniData::getDebugGeometryPoolInitialCapacity() const {
    const auto data = UsdUtil::getCesiumData(_path);

    uint64_t geometryPoolInitialCapacity;
    data.GetDebugGeometryPoolInitialCapacityAttr().Get(&geometryPoolInitialCapacity);

    return geometryPoolInitialCapacity;
}

uint64_t OmniData::getDebugMaterialPoolInitialCapacity() const {
    const auto data = UsdUtil::getCesiumData(_path);

    uint64_t materialPoolInitialCapacity;
    data.GetDebugMaterialPoolInitialCapacityAttr().Get(&materialPoolInitialCapacity);

    return materialPoolInitialCapacity;
}

uint64_t OmniData::getDebugTexturePoolInitialCapacity() const {
    const auto data = UsdUtil::getCesiumData(_path);

    uint64_t texturePoolInitialCapacity;
    data.GetDebugTexturePoolInitialCapacityAttr().Get(&texturePoolInitialCapacity);

    return texturePoolInitialCapacity;
}

bool OmniData::getDebugRandomColors() const {
    const auto data = UsdUtil::getCesiumData(_path);

    bool debugRandomColors;
    data.GetDebugRandomColorsAttr().Get(&debugRandomColors);

    return debugRandomColors;
}

} // namespace cesium::omniverse
