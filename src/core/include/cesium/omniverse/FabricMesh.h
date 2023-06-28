#pragma once

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDefinition.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"

#include <memory>

namespace cesium::omniverse {

class FabricGeometryDefinition;
class FabricMaterialDefinition;

class FabricMesh {
  public:
    FabricMesh(std::shared_ptr<FabricGeometry> geometry, std::shared_ptr<FabricMaterial> material);

    [[nodiscard]] std::shared_ptr<FabricGeometry> getGeometry() const;
    [[nodiscard]] std::shared_ptr<FabricMaterial> getMaterial() const;

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
    std::shared_ptr<FabricGeometry> _geometry;
    std::shared_ptr<FabricMaterial> _material;
};

} // namespace cesium::omniverse
