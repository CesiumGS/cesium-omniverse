#pragma once

#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MetadataUtil.h"

#include <glm/glm.hpp>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/path.h>

#include <vector>

namespace cesium::omniverse {

class FabricMaterialDefinition {
  public:
    FabricMaterialDefinition(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const MaterialInfo& materialInfo,
        const FeaturesInfo& featuresInfo,
        const ImageryLayersInfo& imageryLayersInfo,
        bool disableTextures,
        const pxr::SdfPath& tilesetMaterialPath);

    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] const std::vector<FeatureIdType>& getFeatureIdTypes() const;
    [[nodiscard]] const std::vector<OverlayType>& getImageryOverlayTypes() const;
    [[nodiscard]] const std::vector<OverlayRenderPipe>& getImageryOverlayRenderPipes() const;
    [[nodiscard]] uint64_t getImageryLayerCount() const;
    [[nodiscard]] uint64_t getPolygonImageryCount() const;
    [[nodiscard]] uint64_t getIonImageryCount() const;
    [[nodiscard]] std::vector<int> getIonImageryLayerIndices() const;
    [[nodiscard]] std::vector<int> getPolygonImageryLayerIndices() const;
    [[nodiscard]] bool hasTilesetMaterial() const;
    [[nodiscard]] const pxr::SdfPath& getTilesetMaterialPath() const;
    [[nodiscard]] const std::vector<MetadataUtil::PropertyDefinition>& getProperties() const;

    // Make sure to update this function when adding new fields to the class
    bool operator==(const FabricMaterialDefinition& other) const;

  private:
    bool _hasVertexColors;
    bool _hasBaseColorTexture;
    std::vector<FeatureIdType> _featureIdTypes;
    std::vector<OverlayType> _imageryOverlayTypes;
    std::vector<OverlayRenderPipe> _imageryOverlayRenderPipes;
    pxr::SdfPath _tilesetMaterialPath;
    std::vector<MetadataUtil::PropertyDefinition> _properties;
};

} // namespace cesium::omniverse
