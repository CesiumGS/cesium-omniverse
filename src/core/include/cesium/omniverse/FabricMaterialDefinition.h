#pragma once

#include "cesium/omniverse/GltfUtil.h"

#include <glm/glm.hpp>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {

class FabricMaterialDefinition {
  public:
    FabricMaterialDefinition(
        const MaterialInfo& materialInfo,
        const FeaturesInfo& featuresInfo,
        uint64_t imageryLayerCount,
        bool disableTextures,
        const pxr::SdfPath& tilesetMaterialPath);

    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] const std::vector<FeatureIdType>& getFeatureIdTypes() const;
    [[nodiscard]] uint64_t getFeatureIdIndexCount() const;
    [[nodiscard]] uint64_t getFeatureIdAttributeCount() const;
    [[nodiscard]] uint64_t getFeatureIdTextureCount() const;
    [[nodiscard]] uint64_t getImageryLayerCount() const;
    [[nodiscard]] bool hasTilesetMaterial() const;
    [[nodiscard]] const pxr::SdfPath& getTilesetMaterialPath() const;

    // Make sure to update this function when adding new fields to the class
    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasVertexColors;
    bool _hasBaseColorTexture;
    std::vector<FeatureIdType> _featureIdTypes;
    uint64_t _imageryLayerCount;
    pxr::SdfPath _tilesetMaterialPath;
};

} // namespace cesium::omniverse
