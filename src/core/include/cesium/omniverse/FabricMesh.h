#pragma once

#include "cesium/omniverse/FabricFeaturesInfo.h"
#include "cesium/omniverse/FabricMaterialInfo.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace cesium::omniverse {

class FabricGeometry;
class FabricMaterial;
class FabricTexture;

struct FabricMesh {
    FabricMesh() = default;
    ~FabricMesh() = default;
    FabricMesh(const FabricMesh&) = delete;
    FabricMesh& operator=(const FabricMesh&) = delete;
    FabricMesh(FabricMesh&&) noexcept = default;
    FabricMesh& operator=(FabricMesh&&) noexcept = default;

    std::shared_ptr<FabricGeometry> pGeometry;
    std::shared_ptr<FabricMaterial> pMaterial;
    std::shared_ptr<FabricTexture> pBaseColorTexture;
    std::vector<std::shared_ptr<FabricTexture>> featureIdTextures;
    std::vector<std::shared_ptr<FabricTexture>> propertyTextures;
    std::vector<std::shared_ptr<FabricTexture>> propertyTableTextures;
    FabricMaterialInfo materialInfo;
    FabricFeaturesInfo featuresInfo;
    std::unordered_map<uint64_t, uint64_t> texcoordIndexMapping;
    std::unordered_map<uint64_t, uint64_t> imageryTexcoordIndexMapping;
    std::vector<uint64_t> featureIdIndexSetIndexMapping;
    std::vector<uint64_t> featureIdAttributeSetIndexMapping;
    std::vector<uint64_t> featureIdTextureSetIndexMapping;
    std::unordered_map<uint64_t, uint64_t> propertyTextureIndexMapping;
};

} // namespace cesium::omniverse
