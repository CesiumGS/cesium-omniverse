#pragma once

#include "cesium/omniverse/FabricGeometryDefinition.h"

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class FabricMaterial;

class FabricGeometry {
  public:
    FabricGeometry(pxr::SdfPath path, const FabricGeometryDefinition& geometryDefinition);
    ~FabricGeometry();

    void setTile(
        int64_t tilesetId,
        int64_t tileId,
        const glm::dmat4& ecefToUsdTransform,
        const glm::dmat4& gltfToEcefTransform,
        const glm::dmat4& nodeTransform,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        bool hasImagery,
        const glm::dvec2& imageryTexcoordTranslation,
        const glm::dvec2& imageryTexcoordScale,
        uint64_t imageryTexcoordSetIndex);

    void setActive(bool active);
    void setVisibility(bool visible);

    pxr::SdfPath getPath() const;
    const FabricGeometryDefinition& getGeometryDefinition() const;

    void assignMaterial(std::shared_ptr<FabricMaterial> material);

  private:
    void initialize();
    void reset();

    const pxr::SdfPath _path;
    const FabricGeometryDefinition _geometryDefinition;
};

} // namespace cesium::omniverse
