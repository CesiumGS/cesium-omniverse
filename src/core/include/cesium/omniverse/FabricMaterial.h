#pragma once

#include "cesium/omniverse/FabricMaterialDescriptor.h"
#include "cesium/omniverse/FabricMaterialInfo.h"

#include <glm/glm.hpp>
#include <omni/fabric/IPath.h>

#include <unordered_map>

namespace omni::fabric {
struct Type;
}

namespace omni::ui {
class DynamicTextureProvider;
}

namespace cesium::omniverse {

class FabricTexture;
enum class MdlInternalPropertyType;
struct FabricPropertyDescriptor;
struct FabricTextureInfo;

class FabricMaterial {
  public:
    FabricMaterial(
        Context* pContext,
        const omni::fabric::Path& path,
        const FabricMaterialDescriptor& materialDescriptor,
        const pxr::TfToken& defaultWhiteTextureAssetPathToken,
        const pxr::TfToken& defaultTransparentTextureAssetPathToken,
        bool debugRandomColors,
        int64_t poolId);
    ~FabricMaterial();
    FabricMaterial(const FabricMaterial&) = delete;
    FabricMaterial& operator=(const FabricMaterial&) = delete;
    FabricMaterial(FabricMaterial&&) noexcept = default;
    FabricMaterial& operator=(FabricMaterial&&) noexcept = default;

    void setMaterial(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        int64_t tilesetId,
        const FabricMaterialInfo& materialInfo,
        const FabricFeaturesInfo& featuresInfo,
        FabricTexture* pBaseColorTexture,
        const std::vector<std::shared_ptr<FabricTexture>>& featureIdTextures,
        const std::vector<std::shared_ptr<FabricTexture>>& propertyTextures,
        const std::vector<std::shared_ptr<FabricTexture>>& propertyTableTextures,
        const glm::dvec3& displayColor,
        double displayOpacity,
        const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
        const std::vector<uint64_t>& featureIdIndexSetIndexMapping,
        const std::vector<uint64_t>& featureIdAttributeSetIndexMapping,
        const std::vector<uint64_t>& featureIdTextureSetIndexMapping,
        const std::unordered_map<uint64_t, uint64_t>& propertyTextureIndexMapping);

    void setRasterOverlayLayer(
        FabricTexture* pTexture,
        const FabricTextureInfo& textureInfo,
        uint64_t rasterOverlayLayerIndex,
        double alpha,
        const std::unordered_map<uint64_t, uint64_t>& rasterOverlayTexcoordIndexMapping);

    void setRasterOverlayLayerAlpha(uint64_t rasterOverlayLayerIndex, double alpha);
    void setDisplayColorAndOpacity(const glm::dvec3& displayColor, double displayOpacity);
    void updateShaderInput(const omni::fabric::Path& shaderPath, const omni::fabric::Token& attributeName);
    void clearRasterOverlayLayer(uint64_t rasterOverlayLayerIndex);
    void setActive(bool active);

    [[nodiscard]] const omni::fabric::Path& getPath() const;
    [[nodiscard]] const FabricMaterialDescriptor& getMaterialDescriptor() const;
    [[nodiscard]] int64_t getPoolId() const;

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
    void createRasterOverlayLayer(const omni::fabric::Path& path);
    void createRasterOverlayLayerResolverCommon(
        const omni::fabric::Path& path,
        uint64_t textureCount,
        const omni::fabric::Token& subidentifier);
    void createRasterOverlayLayerResolver(const omni::fabric::Path& path, uint64_t textureCount);
    void createClippingRasterOverlayLayerResolver(const omni::fabric::Path& path, uint64_t textureCount);
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
    void createPropertyAttributeProperty(const omni::fabric::Path& path, MdlInternalPropertyType type);
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
    void createPropertyTextureProperty(const omni::fabric::Path& path, MdlInternalPropertyType type);
    void createPropertyTablePropertyInt(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType);
    void createPropertyTablePropertyNormalizedInt(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType,
        const omni::fabric::Type& offsetType,
        const omni::fabric::Type& scaleType,
        const omni::fabric::Type& maximumValueType);
    void createPropertyTablePropertyFloat(
        const omni::fabric::Path& path,
        const omni::fabric::Token& subidentifier,
        const omni::fabric::Type& noDataType,
        const omni::fabric::Type& defaultValueType,
        const omni::fabric::Type& offsetType,
        const omni::fabric::Type& scaleType);
    void createPropertyTableProperty(const omni::fabric::Path& path, MdlInternalPropertyType type);

    void reset();

    void setShaderValues(
        const omni::fabric::Path& path,
        const FabricMaterialInfo& materialInfo,
        const glm::dvec3& displayColor,
        double displayOpacity);
    void setTextureValues(
        const omni::fabric::Path& path,
        const pxr::TfToken& textureAssetPathToken,
        const FabricTextureInfo& textureInfo,
        uint64_t texcoordIndex);
    void setRasterOverlayLayerValues(
        const omni::fabric::Path& path,
        const pxr::TfToken& textureAssetPathToken,
        const FabricTextureInfo& textureInfo,
        uint64_t texcoordIndex,
        double alpha);
    void setRasterOverlayLayerAlphaValue(const omni::fabric::Path& path, double alpha);
    void setFeatureIdIndexValues(const omni::fabric::Path& path, int nullFeatureId);
    void setFeatureIdAttributeValues(const omni::fabric::Path& path, const std::string& primvarName, int nullFeatureId);
    void setFeatureIdTextureValues(
        const omni::fabric::Path& path,
        const pxr::TfToken& textureAssetPathToken,
        const FabricTextureInfo& textureInfo,
        uint64_t texcoordIndex,
        int nullFeatureId);

    void createConnectionsToCopiedPaths();
    void destroyConnectionsToCopiedPaths();
    void createConnectionsToProperties();
    void destroyConnectionsToProperties();

    bool stageDestroyed();

    Context* _pContext;
    omni::fabric::Path _materialPath;
    FabricMaterialDescriptor _materialDescriptor;
    pxr::TfToken _defaultWhiteTextureAssetPathToken;
    pxr::TfToken _defaultTransparentTextureAssetPathToken;
    bool _debugRandomColors;
    int64_t _poolId;
    int64_t _stageId;
    bool _usesDefaultMaterial;

    FabricAlphaMode _alphaMode{FabricAlphaMode::OPAQUE};
    glm::dvec3 _debugColor{1.0, 1.0, 1.0};

    omni::fabric::Path _shaderPath;
    omni::fabric::Path _baseColorTexturePath;

    std::vector<omni::fabric::Path> _rasterOverlayLayerPaths;
    omni::fabric::Path _overlayRasterOverlayLayerResolverPath;
    omni::fabric::Path _clippingRasterOverlayLayerResolverPath;

    std::vector<omni::fabric::Path> _featureIdPaths;
    std::vector<omni::fabric::Path> _featureIdIndexPaths;
    std::vector<omni::fabric::Path> _featureIdAttributePaths;
    std::vector<omni::fabric::Path> _featureIdTexturePaths;

    std::vector<omni::fabric::Path> _propertyPaths;
    std::unordered_map<MdlInternalPropertyType, std::vector<omni::fabric::Path>> _propertyAttributePropertyPaths;
    std::unordered_map<MdlInternalPropertyType, std::vector<omni::fabric::Path>> _propertyTexturePropertyPaths;
    std::unordered_map<MdlInternalPropertyType, std::vector<omni::fabric::Path>> _propertyTablePropertyPaths;

    std::vector<omni::fabric::Path> _copiedBaseColorTexturePaths;
    std::vector<omni::fabric::Path> _copiedRasterOverlayLayerPaths;
    std::vector<omni::fabric::Path> _copiedFeatureIdPaths;
    std::vector<omni::fabric::Path> _copiedPropertyPaths;

    std::vector<omni::fabric::Path> _allPaths;
};

} // namespace cesium::omniverse
