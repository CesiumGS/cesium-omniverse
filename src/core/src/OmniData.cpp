#include "cesium/omniverse/OmniData.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/data.h>

namespace cesium::omniverse {

OmniData::OmniData(Context* pContext, const PXR_NS::SdfPath& path)
    : _pContext(pContext)
    , _path(path) {}

const PXR_NS::SdfPath& OmniData::getPath() const {
    return _path;
}

PXR_NS::SdfPath OmniData::getSelectedIonServerPath() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    PXR_NS::SdfPathVector targets;
    cesiumData.GetSelectedIonServerRel().GetForwardedTargets(&targets);

    if (targets.empty()) {
        return {};
    }

    return targets.front();
}

bool OmniData::getDebugDisableMaterials() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool disableMaterials;
    cesiumData.GetDebugDisableMaterialsAttr().Get(&disableMaterials);

    return disableMaterials;
}

bool OmniData::getDebugDisableTextures() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool disableTextures;
    cesiumData.GetDebugDisableTexturesAttr().Get(&disableTextures);

    return disableTextures;
}

bool OmniData::getDebugDisableGeometryPool() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool disableGeometryPool;
    cesiumData.GetDebugDisableGeometryPoolAttr().Get(&disableGeometryPool);

    return disableGeometryPool;
}

bool OmniData::getDebugDisableMaterialPool() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool disableMaterialPool;
    cesiumData.GetDebugDisableMaterialPoolAttr().Get(&disableMaterialPool);

    return disableMaterialPool;
}

bool OmniData::getDebugDisableTexturePool() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool disableTexturePool;
    cesiumData.GetDebugDisableTexturePoolAttr().Get(&disableTexturePool);

    return disableTexturePool;
}

uint64_t OmniData::getDebugGeometryPoolInitialCapacity() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    uint64_t geometryPoolInitialCapacity;
    cesiumData.GetDebugGeometryPoolInitialCapacityAttr().Get(&geometryPoolInitialCapacity);

    return geometryPoolInitialCapacity;
}

uint64_t OmniData::getDebugMaterialPoolInitialCapacity() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    uint64_t materialPoolInitialCapacity;
    cesiumData.GetDebugMaterialPoolInitialCapacityAttr().Get(&materialPoolInitialCapacity);

    return materialPoolInitialCapacity;
}

uint64_t OmniData::getDebugTexturePoolInitialCapacity() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    uint64_t texturePoolInitialCapacity;
    cesiumData.GetDebugTexturePoolInitialCapacityAttr().Get(&texturePoolInitialCapacity);

    return texturePoolInitialCapacity;
}

bool OmniData::getDebugRandomColors() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool debugRandomColors;
    cesiumData.GetDebugRandomColorsAttr().Get(&debugRandomColors);

    return debugRandomColors;
}

bool OmniData::getDebugDisableGeoreferencing() const {
    const auto cesiumData = UsdUtil::getCesiumData(_pContext->getUsdStage(), _path);

    bool debugDisableGeoreferencing;
    cesiumData.GetDebugDisableGeoreferencingAttr().Get(&debugDisableGeoreferencing);

    return debugDisableGeoreferencing;
}

} // namespace cesium::omniverse
