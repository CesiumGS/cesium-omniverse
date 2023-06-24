#pragma once

#include "cesium/omniverse/OmniGeometry.h"
#include "cesium/omniverse/OmniGeometryDefinition.h"

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/mesh.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class OmniMaterial;

class UsdGeometry final : public OmniGeometry {
  public:
    UsdGeometry(pxr::SdfPath path, const OmniGeometryDefinition& geometryDefinition, bool debugRandomColors);
    ~UsdGeometry();

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
        uint64_t imageryTexcoordSetIndex) override;

    void setVisibility(bool visible) override;

    void assignMaterial(std::shared_ptr<OmniMaterial> material) override;

  private:
    void initialize();
    void reset() override;

    pxr::UsdGeomMesh _mesh;
};

} // namespace cesium::omniverse
