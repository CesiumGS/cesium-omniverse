#pragma once

#include "cesium/omniverse/FabricGeometryDescriptor.h"

#include <glm/fwd.hpp>
#include <omni/fabric/IPath.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class Context;
struct FabricMaterialInfo;

class FabricGeometry {
  public:
    FabricGeometry(
        Context* pContext,
        const omni::fabric::Path& path,
        const FabricGeometryDescriptor& geometryDescriptor,
        int64_t poolId);
    ~FabricGeometry();
    FabricGeometry(const FabricGeometry&) = delete;
    FabricGeometry& operator=(const FabricGeometry&) = delete;
    FabricGeometry(FabricGeometry&&) noexcept = default;
    FabricGeometry& operator=(FabricGeometry&&) noexcept = default;

    void setGeometry(
        int64_t tilesetId,
        const glm::dmat4& ecefToPrimWorldTransform,
        const glm::dmat4& gltfLocalToEcefTransform,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const FabricMaterialInfo& materialInfo,
        bool smoothNormals,
        const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
        const std::unordered_map<uint64_t, uint64_t>& imageryTexcoordIndexMapping);

    void setActive(bool active);
    void setVisibility(bool visible);

    [[nodiscard]] const omni::fabric::Path& getPath() const;
    [[nodiscard]] const FabricGeometryDescriptor& getGeometryDescriptor() const;
    [[nodiscard]] int64_t getPoolId() const;

    void setMaterial(const omni::fabric::Path& materialPath);

  private:
    void initialize();
    void reset();
    bool stageDestroyed();

    Context* _pContext;
    omni::fabric::Path _path;
    FabricGeometryDescriptor _geometryDescriptor;
    int64_t _poolId;
    int64_t _stageId;
};

} // namespace cesium::omniverse
