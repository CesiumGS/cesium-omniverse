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
    FabricGeometry(const omni::fabric::Path& path, const FabricGeometryDefinition& geometryDefinition, long stageId);
    ~FabricGeometry();

    void setGeometry(
        int64_t tilesetId,
        const glm::dmat4& ecefToUsdTransform,
        const glm::dmat4& gltfToEcefTransform,
        const glm::dmat4& nodeTransform,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
        const std::unordered_map<uint64_t, uint64_t>& imageryTexcoordIndexMapping);

    void setActive(bool active);
    void setVisibility(bool visible);

    [[nodiscard]] const omni::fabric::Path& getPath() const;
    [[nodiscard]] const FabricGeometryDefinition& getGeometryDefinition() const;

    void setMaterial(const omni::fabric::Path& materialPath);

  private:
    void initialize();
    void reset();
    bool stageDestroyed();

    const omni::fabric::Path _path;
    const FabricGeometryDefinition _geometryDefinition;
    const long _stageId;
};

} // namespace cesium::omniverse
