#pragma once

#include "cesium/omniverse/OmniGeometryDefinition.h"

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class OmniMaterial;

class OmniGeometry {
  public:
    OmniGeometry(pxr::SdfPath path, const OmniGeometryDefinition& geometryDefinition, bool debugRandomColors);
    virtual ~OmniGeometry() = default;

    virtual void setTile(
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
        uint64_t imageryTexcoordSetIndex) = 0;

    void setActive(bool active);
    virtual void setVisibility(bool visible) = 0;

    pxr::SdfPath getPath() const;
    const OmniGeometryDefinition& getGeometryDefinition() const;

    virtual void assignMaterial(std::shared_ptr<OmniMaterial> material) = 0;

  protected:
    virtual void reset() = 0;

    const pxr::SdfPath _path;
    const OmniGeometryDefinition _geometryDefinition;
    const bool _debugRandomColors;
};

} // namespace cesium::omniverse
