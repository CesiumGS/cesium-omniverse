#include "cesium/omniverse/OmniMaterial.h"

namespace cesium::omniverse {

OmniMaterial::OmniMaterial(pxr::SdfPath path, const OmniMaterialDefinition& materialDefinition)
    : _path(path)
    , _materialDefinition(materialDefinition) {}

void OmniMaterial::setActive(bool active) {
    if (!active) {
        reset();
    }
}

pxr::SdfPath OmniMaterial::getPath() const {
    return _path;
}

const OmniMaterialDefinition& OmniMaterial::getMaterialDefinition() const {
    return _materialDefinition;
}

}; // namespace cesium::omniverse
