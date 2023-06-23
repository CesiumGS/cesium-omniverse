#pragma once

#include "cesium/omniverse/OmniGeometry.h"
#include "cesium/omniverse/OmniGeometryDefinition.h"
#include "cesium/omniverse/OmniMaterial.h"
#include "cesium/omniverse/OmniMaterialDefinition.h"

#include <memory>

namespace cesium::omniverse {

class OmniGeometryDefinition;
class OmniMaterialDefinition;

class OmniMesh {
  public:
    OmniMesh(std::shared_ptr<OmniGeometry> geometry, std::shared_ptr<OmniMaterial> material);

    std::shared_ptr<OmniGeometry> getGeometry() const;
    std::shared_ptr<OmniMaterial> getMaterial() const;

    void setVisibility(bool visible) const;

    void setTile(
        int64_t tilesetId,
        int64_t tileId,
        const glm::dmat4& ecefToUsdTransform,
        const glm::dmat4& gltfToEcefTransform,
        const glm::dmat4& nodeTransform,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        const CesiumGltf::ImageCesium* imagery,
        const glm::dvec2& imageryTexcoordTranslation,
        const glm::dvec2& imageryTexcoordScale,
        uint64_t imageryTexcoordSetIndex);

  private:
    std::shared_ptr<OmniGeometry> _geometry;
    std::shared_ptr<OmniMaterial> _material;
};

} // namespace cesium::omniverse
