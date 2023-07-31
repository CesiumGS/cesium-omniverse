#pragma once

#include "cesium/omniverse/FabricGeometryDefinition.h"

#include <glm/glm.hpp>
#include <omni/fabric/IPath.h>
#include <pxr/usd/sdf/path.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class FabricGeometry {
  public:
    FabricGeometry(
        const pxr::SdfPath& path,
        const FabricGeometryDefinition& geometryDefinition,
        bool debugRandomColors);
    ~FabricGeometry();

    void setGeometry(
        int64_t tilesetId,
        const glm::dmat4& ecefToUsdTransform,
        const glm::dmat4& gltfToEcefTransform,
        const glm::dmat4& nodeTransform,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        bool hasImagery);

    void setActive(bool active);
    void setVisibility(bool visible);

    [[nodiscard]] omni::fabric::Path getPathFabric() const;
    [[nodiscard]] const FabricGeometryDefinition& getGeometryDefinition() const;

    void setMaterial(const omni::fabric::Path& materialPath);

  private:
    void initialize();
    void reset();

    const omni::fabric::Path _pathFabric;
    const FabricGeometryDefinition _geometryDefinition;
    const bool _debugRandomColors;
};

} // namespace cesium::omniverse
