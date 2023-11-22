#pragma once

#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <omni/fabric/IPath.h>
#include <omni/fabric/Type.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>

namespace omni::ui {
class DynamicTextureProvider;
}

namespace cesium::omniverse {

class FabricTexture;

class FabricMaterial {
  public:
    FabricMaterial(
        const omni::fabric::Path& path,
        const FabricMaterialDefinition& materialDefinition,
        const pxr::TfToken& defaultTextureAssetPathToken,
        const pxr::TfToken& defaultTransparentTextureAssetPathToken,
        bool debugRandomColors,
        long stageId);
    ~FabricMaterial();

    void setMaterial(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        int64_t tilesetId,
        const MaterialInfo& materialInfo,
        const FeaturesInfo& featuresInfo,
        const std::shared_ptr<FabricTexture>& baseColorTexture,
        const std::vector<std::shared_ptr<FabricTexture>>& featureIdTextures,
        const std::vector<std::shared_ptr<FabricTexture>>& propertyTextures,
        const glm::dvec3& displayColor,
        double displayOpacity,
        const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
        const std::vector<uint64_t>& featureIdIndexSetIndexMapping,
        const std::vector<uint64_t>& featureIdAttributeSetIndexMapping,
        const std::vector<uint64_t>& featureIdTextureSetIndexMapping,
        const std::unordered_map<uint64_t, uint64_t>& propertyTextureIndexMapping);

    void setImageryLayer(
        const std::shared_ptr<FabricTexture>& texture,
        const TextureInfo& textureInfo,
        uint64_t imageryLayerIndex,
        double alpha,
        const std::unordered_map<uint64_t, uint64_t>& imageryTexcoordIndexMapping);

    void setImageryLayerAlpha(uint64_t imageryLayerIndex, double alpha);
    void setDisplayColorAndOpacity(const glm::dvec3& displayColor, double displayOpacity);
    void updateShaderInput(const omni::fabric::Path& shaderPath, const omni::fabric::Token& attributeName);
    void clearImageryLayer(uint64_t imageryLayerIndex);
    void setActive(bool active);

    [[nodiscard]] const omni::fabric::Path& getPath() const;
    [[nodiscard]] const FabricMaterialDefinition& getMaterialDefinition() const;

  private:
    void initializeNodes();
    void initializeDefaultMaterial();
    void initializeExistingMaterial(const omni::fabric::Path& path);

    void createMaterial(const omni::fabric::Path& path);
    void createShader(const omni::fabric::Path& path);
    void createTextureCommon(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subIdentifier,
        const std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>& additionalAttributes = {});
    void createTexture(const omni::fabric::Path& path);
    void createImageryLayer(const omni::fabric::Path& path);
    void createImageryLayerResolver(const omni::fabric::Path& path, uint64_t textureCount);
    void createFeatureIdIndex(const omni::fabric::Path& path);
    void createFeatureIdAttribute(const omni::fabric::Path& path);
    void createFeatureIdTexture(const omni::fabric::Path& path);
    void createPropertyAttributePropertyInt(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType);
    void createPropertyAttributePropertyNormalizedInt(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType,
        const omni::fabric::Type& offsetType,
        const omni::fabric::Type& scaleType,
        const omni::fabric::Type& maximumValueType);
    void createPropertyAttributePropertyFloat(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType,
        const omni::fabric::Type& offsetType,
        const omni::fabric::Type& scaleType);
    void createPropertyTexturePropertyInt(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType);
    void createPropertyTexturePropertyNormalizedInt(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType,
        const omni::fabric::Type& offsetType,
        const omni::fabric::Type& scaleType,
        const omni::fabric::Type& maximumValueType);
    void createPropertyAttributeProperty(const omni::fabric::Path& path, DataType type);
    void createPropertyTextureProperty(const omni::fabric::Path& path, DataType type);

    void reset();

    void setShaderValues(
        const omni::fabric::Path& path,
        const MaterialInfo& materialInfo,
        const glm::dvec3& displayColor,
        double displayOpacity);
    void setTextureValues(
        const omni::fabric::Path& path,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex);
    void setImageryLayerValues(
        const omni::fabric::Path& path,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex,
        double alpha);
    void setImageryLayerAlphaValue(const omni::fabric::Path& path, double alpha);
    void setFeatureIdIndexValues(const omni::fabric::Path& path, int nullFeatureId);
    void setFeatureIdAttributeValues(const omni::fabric::Path& path, const std::string& primvarName, int nullFeatureId);
    void setFeatureIdTextureValues(
        const omni::fabric::Path& path,
        const pxr::TfToken& textureAssetPathToken,
        const TextureInfo& textureInfo,
        uint64_t texcoordIndex,
        int nullFeatureId);

    void createConnectionsToCopiedPaths();
    void destroyConnectionsToCopiedPaths();

    bool stageDestroyed();

    omni::fabric::Path _materialPath;
    const FabricMaterialDefinition _materialDefinition;
    const pxr::TfToken _defaultTextureAssetPathToken;
    const pxr::TfToken _defaultTransparentTextureAssetPathToken;
    const bool _debugRandomColors;
    const long _stageId;

    bool _usesDefaultMaterial;

    AlphaMode _alphaMode{AlphaMode::OPAQUE};
    glm::dvec3 _debugColor{1.0, 1.0, 1.0};

    omni::fabric::Path _shaderPath;
    omni::fabric::Path _baseColorTexturePath;
    std::vector<omni::fabric::Path> _imageryLayerPaths;
    omni::fabric::Path _imageryLayerResolverPath;
    std::vector<omni::fabric::Path> _featureIdPaths;
    std::vector<omni::fabric::Path> _featureIdIndexPaths;
    std::vector<omni::fabric::Path> _featureIdAttributePaths;
    std::vector<omni::fabric::Path> _featureIdTexturePaths;
    std::unordered_map<DataType, std::vector<omni::fabric::Path>> _propertyAttributePropertyPaths;
    std::unordered_map<DataType, std::vector<omni::fabric::Path>> _propertyTexturePropertyPaths;

    std::vector<omni::fabric::Path> _copiedBaseColorTexturePaths;
    std::vector<omni::fabric::Path> _copiedImageryLayerPaths;
    std::vector<omni::fabric::Path> _copiedFeatureIdPaths;

    std::vector<omni::fabric::Path> _allPaths;
};

} // namespace cesium::omniverse
