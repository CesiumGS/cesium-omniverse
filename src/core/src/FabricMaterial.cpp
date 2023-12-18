#include "cesium/omniverse/FabricMaterial.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/gtc/random.hpp>
#include <omni/fabric/FabricUSD.h>
#include <spdlog/fmt/fmt.h>
#include <stdint.h>

#include <iostream>

namespace cesium::omniverse {

namespace {

// Should match MAX_IMAGERY_LAYERS_COUNT in cesium.mdl
const uint64_t MAX_IMAGERY_LAYERS_COUNT = 16;

const auto DEFAULT_DEBUG_COLOR = glm::dvec3(1.0, 1.0, 1.0);
const auto DEFAULT_ALPHA = 1.0f;
const auto DEFAULT_DISPLAY_COLOR = glm::dvec3(1.0, 1.0, 1.0);
const auto DEFAULT_DISPLAY_OPACITY = 1.0;
const auto DEFAULT_TEXCOORD_INDEX = uint64_t(0);
const auto DEFAULT_FEATURE_ID_PRIMVAR_NAME = std::string("_FEATURE_ID_0");
const auto DEFAULT_NULL_FEATURE_ID = -1;

struct FeatureIdCounts {
    uint64_t indexCount;
    uint64_t attributeCount;
    uint64_t textureCount;
    uint64_t totalCount;
};

FeatureIdCounts getFeatureIdCounts(const FabricMaterialDefinition& materialDefinition) {
    const auto& featureIdTypes = materialDefinition.getFeatureIdTypes();
    auto featureIdCount = featureIdTypes.size();

    uint64_t indexCount = 0;
    uint64_t attributeCount = 0;
    uint64_t textureCount = 0;

    for (uint64_t i = 0; i < featureIdCount; i++) {
        const auto featureIdType = featureIdTypes[i];
        switch (featureIdType) {
            case FeatureIdType::INDEX:
                indexCount++;
                break;
            case FeatureIdType::ATTRIBUTE:
                attributeCount++;
                break;
            case FeatureIdType::TEXTURE:
                textureCount++;
                break;
        }
    }

    return FeatureIdCounts{indexCount, attributeCount, textureCount, featureIdCount};
}

uint64_t getImageryLayerCount(const FabricMaterialDefinition& materialDefinition) {
    auto imageryLayerCount = materialDefinition.getImageryLayerCount();

    if (imageryLayerCount > MAX_IMAGERY_LAYERS_COUNT) {
        CESIUM_LOG_WARN(
            "Number of imagery layers ({}) exceeds maximum imagery layer count ({}). Excess imagery layers will be "
            "ignored.",
            imageryLayerCount,
            MAX_IMAGERY_LAYERS_COUNT);
    }

    imageryLayerCount = glm::min(imageryLayerCount, MAX_IMAGERY_LAYERS_COUNT);

    return imageryLayerCount;
}

int getAlphaMode(AlphaMode alphaMode, double displayOpacity) {
    return static_cast<int>(displayOpacity < 1.0 ? AlphaMode::BLEND : alphaMode);
}

pxr::GfVec4f getTileColor(const glm::dvec3& debugColor, const glm::dvec3& displayColor, double displayOpacity) {
    const auto finalColor = glm::dvec4(debugColor * displayColor, displayOpacity);
    return {
        static_cast<float>(finalColor.x),
        static_cast<float>(finalColor.y),
        static_cast<float>(finalColor.z),
        static_cast<float>(finalColor.w),
    };
}

void createConnection(
    omni::fabric::StageReaderWriter& srw,
    const omni::fabric::Path& outputPath,
    const omni::fabric::Path& inputPath,
    const omni::fabric::Token& inputName) {
    srw.createConnection(inputPath, inputName, omni::fabric::Connection{outputPath, FabricTokens::outputs_out});
}

void destroyConnection(
    omni::fabric::StageReaderWriter& srw,
    const omni::fabric::Path& inputPath,
    const omni::fabric::Token& inputName) {
    srw.destroyConnection(inputPath, inputName);
}

template <typename T> const T& defaultValue(const T* value, const T& defaultValue) {
    return value == nullptr ? defaultValue : *value;
}

template <typename T, typename U> T defaultValue(const std::optional<U>& optional, const T& defaultValue) {
    return optional.has_value() ? static_cast<T>(optional.value()) : defaultValue;
}

void createAttributes(
    omni::fabric::StageReaderWriter& srw,
    const omni::fabric::Path& path,
    FabricAttributesBuilder& attributes,
    const omni::fabric::Token& subidentifier) {

    // clang-format off
    attributes.addAttribute(FabricTypes::inputs_excludeFromWhiteMode, FabricTokens::inputs_excludeFromWhiteMode);
    attributes.addAttribute(FabricTypes::outputs_out, FabricTokens::outputs_out);
    attributes.addAttribute(FabricTypes::info_implementationSource, FabricTokens::info_implementationSource);
    attributes.addAttribute(FabricTypes::info_mdl_sourceAsset, FabricTokens::info_mdl_sourceAsset);
    attributes.addAttribute(FabricTypes::info_mdl_sourceAsset_subIdentifier, FabricTokens::info_mdl_sourceAsset_subIdentifier);
    attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
    attributes.addAttribute(FabricTypes::_sdrMetadata, FabricTokens::_sdrMetadata);
    attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
    // clang-format on

    attributes.createAttributes(path);

    // clang-format off
    auto inputsExcludeFromWhiteModeFabric = srw.getAttributeWr<bool>(path, FabricTokens::inputs_excludeFromWhiteMode);
    auto infoImplementationSourceFabric = srw.getAttributeWr<omni::fabric::TokenC>(path, FabricTokens::info_implementationSource);
    auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(path, FabricTokens::info_mdl_sourceAsset);
    auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::TokenC>(path, FabricTokens::info_mdl_sourceAsset_subIdentifier);
    // clang-format on

    srw.setArrayAttributeSize(path, FabricTokens::_paramColorSpace, 0);
    srw.setArrayAttributeSize(path, FabricTokens::_sdrMetadata, 0);

    *inputsExcludeFromWhiteModeFabric = false;
    *infoImplementationSourceFabric = FabricTokens::sourceAsset;
    infoMdlSourceAssetFabric->assetPath = Context::instance().getCesiumMdlPathToken();
    infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
    *infoMdlSourceAssetSubIdentifierFabric = subidentifier;
}

} // namespace

FabricMaterial::FabricMaterial(
    const omni::fabric::Path& path,
    const FabricMaterialDefinition& materialDefinition,
    const pxr::TfToken& defaultTextureAssetPathToken,
    const pxr::TfToken& defaultTransparentTextureAssetPathToken,
    bool debugRandomColors,
    long stageId)
    : _materialPath(path)
    , _materialDefinition(materialDefinition)
    , _defaultTextureAssetPathToken(defaultTextureAssetPathToken)
    , _defaultTransparentTextureAssetPathToken(defaultTransparentTextureAssetPathToken)
    , _debugRandomColors(debugRandomColors)
    , _stageId(stageId)
    , _usesDefaultMaterial(!materialDefinition.hasTilesetMaterial()) {

    if (stageDestroyed()) {
        return;
    }

    initializeNodes();

    if (_usesDefaultMaterial) {
        initializeDefaultMaterial();
    } else {
        const auto existingMaterialPath = FabricUtil::toFabricPath(materialDefinition.getTilesetMaterialPath());
        initializeExistingMaterial(existingMaterialPath);
    }

    for (const auto& nodePath : _allPaths) {
        FabricResourceManager::getInstance().retainPath(nodePath);
    }

    reset();
}

FabricMaterial::~FabricMaterial() {
    if (stageDestroyed()) {
        return;
    }

    for (const auto& path : _allPaths) {
        FabricUtil::destroyPrim(path);
    }
}

void FabricMaterial::setActive(bool active) {
    if (stageDestroyed()) {
        return;
    }

    if (!active) {
        reset();
    }
}

const omni::fabric::Path& FabricMaterial::getPath() const {
    return _materialPath;
}

const FabricMaterialDefinition& FabricMaterial::getMaterialDefinition() const {
    return _materialDefinition;
}

void FabricMaterial::initializeNodes() {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    // Create base color texture
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();
    if (hasBaseColorTexture) {
        const auto baseColorTexturePath = FabricUtil::joinPaths(_materialPath, FabricTokens::base_color_texture);
        createTexture(baseColorTexturePath);
        _baseColorTexturePath = baseColorTexturePath;
        _allPaths.push_back(baseColorTexturePath);
    }

    // Create imagery layers
    const auto imageryLayerCount = getImageryLayerCount(_materialDefinition);
    _imageryLayerPaths.reserve(imageryLayerCount);
    for (uint64_t i = 0; i < imageryLayerCount; i++) {
        const auto imageryLayerPath = FabricUtil::joinPaths(_materialPath, FabricTokens::imagery_layer_n(i));
        createImageryLayer(imageryLayerPath);
        _imageryLayerPaths.push_back(imageryLayerPath);
        _allPaths.push_back(imageryLayerPath);
    }

    // Create feature ids
    const auto& featureIdTypes = _materialDefinition.getFeatureIdTypes();
    const auto featureIdCounts = getFeatureIdCounts(_materialDefinition);
    _featureIdPaths.reserve(featureIdCounts.totalCount);
    _featureIdIndexPaths.reserve(featureIdCounts.indexCount);
    _featureIdAttributePaths.reserve(featureIdCounts.attributeCount);
    _featureIdTexturePaths.reserve(featureIdCounts.textureCount);

    for (uint64_t i = 0; i < featureIdCounts.totalCount; i++) {
        const auto featureIdType = featureIdTypes[i];
        const auto featureIdPath = FabricUtil::joinPaths(_materialPath, FabricTokens::feature_id_n(i));
        switch (featureIdType) {
            case FeatureIdType::INDEX:
                createFeatureIdIndex(featureIdPath);
                _featureIdIndexPaths.push_back(featureIdPath);
                break;
            case FeatureIdType::ATTRIBUTE:
                createFeatureIdAttribute(featureIdPath);
                _featureIdAttributePaths.push_back(featureIdPath);
                break;
            case FeatureIdType::TEXTURE:
                createFeatureIdTexture(featureIdPath);
                _featureIdTexturePaths.push_back(featureIdPath);
                break;
        }
        _featureIdPaths.push_back(featureIdPath);
        _allPaths.push_back(featureIdPath);
    }
}

void FabricMaterial::initializeDefaultMaterial() {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    // const auto imageryLayerCount = getImageryLayerCount(_materialDefinition);
    const uint64_t ionImageryLayerCount = _materialDefinition.getIonImageryCount();
    const uint64_t polygonImageryLayerCount = _materialDefinition.getPolygonImageryCount();
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();


    // Create material
    const auto& materialPath = _materialPath;
    createMaterial(materialPath);
    _allPaths.push_back(materialPath);

    // Create shader
    const auto shaderPath = FabricUtil::joinPaths(materialPath, FabricTokens::cesium_internal_material);
    createShader(shaderPath);
    _shaderPath = shaderPath;
    _allPaths.push_back(shaderPath);

    // Create ion imagery layer resolver if there are multiple ion imagery layers
    if (ionImageryLayerCount > 1) {
        const auto imageryLayerResolverPath = FabricUtil::joinPaths(materialPath, FabricTokens::imagery_layer_resolver);
        createImageryLayerResolver(imageryLayerResolverPath, ionImageryLayerCount);
        _imageryLayerResolverPath = imageryLayerResolverPath;
        _allPaths.push_back(imageryLayerResolverPath);
    }

    // Create polygon imagery layer resolver if there are multiple polygon imagery layers
    // TODO
    // if (polygonImageryLayerCount > 1) {
    //     const auto imageryLayerResolverPath = FabricUtil::joinPaths(materialPath, FabricTokens::imagery_layer_resolver);
    //     createImageryLayerResolver(imageryLayerResolverPath, ionImageryLayerCount);
    //     _imageryLayerResolverPath = imageryLayerResolverPath;
    //     _allPaths.push_back(imageryLayerResolverPath);
    // }


    // Create connection from shader to material
    createConnection(srw, shaderPath, materialPath, FabricTokens::outputs_mdl_surface);
    createConnection(srw, shaderPath, materialPath, FabricTokens::outputs_mdl_displacement);
    createConnection(srw, shaderPath, materialPath, FabricTokens::outputs_mdl_volume);

    // Create connection from base color texture to shader
    if (hasBaseColorTexture) {
        createConnection(srw, _baseColorTexturePath, shaderPath, FabricTokens::inputs_base_color_texture);
    }

    if (ionImageryLayerCount == 1) {
        // Create connection from imagery layer to shader
        const auto& imageryLayerPath = _imageryLayerPaths.front();
        createConnection(srw, imageryLayerPath, shaderPath, FabricTokens::inputs_imagery_layer);
    } else if (ionImageryLayerCount > 1) {
        // Create connection from imagery layer resolver to shader
        createConnection(srw, _imageryLayerResolverPath, shaderPath, FabricTokens::inputs_imagery_layer);

        // Create connections from imagery layers to imagery layer resolver
        for (uint64_t i = 0; i < ionImageryLayerCount; i++) {
            const auto& imageryLayerPath = _imageryLayerPaths[i];
            createConnection(srw, imageryLayerPath, _imageryLayerResolverPath, FabricTokens::inputs_imagery_layer_n(i));
        }
    }


    if (polygonImageryLayerCount == 1) {
        uint64_t polygonStart = ionImageryLayerCount;
        const auto& polygonImageryLayerPath = _imageryLayerPaths[polygonStart];
        // createConnection(srw, polygonImageryLayerPath, shaderPath, FabricTokens::inputs_polygon_imagery_layer);
        createConnection(srw, polygonImageryLayerPath, shaderPath, FabricTokens::inputs_alpha_clip);
    }
    // TODO
    // else if (polygonImageryLayerCount > 1) {
    //     // TODO
    //     std::cout << "polygonImageryCount: " << polygonImageryLayerCount << std::endl;
    // }
}

void FabricMaterial::initializeExistingMaterial(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto copiedPaths = FabricUtil::copyMaterial(path, _materialPath);

    for (const auto& copiedPath : copiedPaths) {
        srw.createAttribute(copiedPath, FabricTokens::_cesium_tilesetId, FabricTypes::_cesium_tilesetId);
        _allPaths.push_back(copiedPath);

        const auto mdlIdentifier = FabricUtil::getMdlIdentifier(copiedPath);

        if (mdlIdentifier == FabricTokens::cesium_base_color_texture_float4) {
            _copiedBaseColorTexturePaths.push_back(copiedPath);
        } else if (mdlIdentifier == FabricTokens::cesium_imagery_layer_float4) {
            _copiedImageryLayerPaths.push_back(copiedPath);
        } else if (mdlIdentifier == FabricTokens::cesium_feature_id_int) {
            _copiedFeatureIdPaths.push_back(copiedPath);
        }
    }

    createConnectionsToCopiedPaths();
}

void FabricMaterial::createMaterial(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStageReaderWriter();
    srw.createPrim(path);

    FabricAttributesBuilder attributes;

    attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);

    attributes.createAttributes(path);
}

void FabricMaterial::createShader(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(path);

    FabricAttributesBuilder attributes;

    attributes.addAttribute(FabricTypes::inputs_tile_color, FabricTokens::inputs_tile_color);
    attributes.addAttribute(FabricTypes::inputs_alpha_cutoff, FabricTokens::inputs_alpha_cutoff);
    attributes.addAttribute(FabricTypes::inputs_alpha_mode, FabricTokens::inputs_alpha_mode);
    attributes.addAttribute(FabricTypes::inputs_base_alpha, FabricTokens::inputs_base_alpha);
    attributes.addAttribute(FabricTypes::inputs_base_color_factor, FabricTokens::inputs_base_color_factor);
    attributes.addAttribute(FabricTypes::inputs_emissive_factor, FabricTokens::inputs_emissive_factor);
    attributes.addAttribute(FabricTypes::inputs_metallic_factor, FabricTokens::inputs_metallic_factor);
    attributes.addAttribute(FabricTypes::inputs_roughness_factor, FabricTokens::inputs_roughness_factor);

    createAttributes(srw, path, attributes, FabricTokens::cesium_internal_material);
}

void FabricMaterial::createTextureCommon(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subIdentifier,
    const std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>& additionalAttributes) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(path);

    FabricAttributesBuilder attributes;

    attributes.addAttribute(FabricTypes::inputs_tex_coord_offset, FabricTokens::inputs_tex_coord_offset);
    attributes.addAttribute(FabricTypes::inputs_tex_coord_rotation, FabricTokens::inputs_tex_coord_rotation);
    attributes.addAttribute(FabricTypes::inputs_tex_coord_scale, FabricTokens::inputs_tex_coord_scale);
    attributes.addAttribute(FabricTypes::inputs_tex_coord_index, FabricTokens::inputs_tex_coord_index);
    attributes.addAttribute(FabricTypes::inputs_texture, FabricTokens::inputs_texture);
    attributes.addAttribute(FabricTypes::inputs_wrap_s, FabricTokens::inputs_wrap_s);
    attributes.addAttribute(FabricTypes::inputs_wrap_t, FabricTokens::inputs_wrap_t);

    for (const auto& additionalAttribute : additionalAttributes) {
        attributes.addAttribute(additionalAttribute.first, additionalAttribute.second);
    }

    createAttributes(srw, path, attributes, subIdentifier);

    // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
    srw.setArrayAttributeSize(path, FabricTokens::_paramColorSpace, 2);
    auto paramColorSpaceFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(path, FabricTokens::_paramColorSpace);
    paramColorSpaceFabric[0] = FabricTokens::inputs_texture;
    paramColorSpaceFabric[1] = FabricTokens::_auto;
}

void FabricMaterial::createTexture(const omni::fabric::Path& path) {
    return createTextureCommon(path, FabricTokens::cesium_internal_texture_lookup);
}

void FabricMaterial::createImageryLayer(const omni::fabric::Path& path) {
    static const auto additionalAttributes = std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>{{
        std::make_pair(FabricTypes::inputs_alpha, FabricTokens::inputs_alpha),
    }};
    return createTextureCommon(path, FabricTokens::cesium_internal_imagery_layer_lookup, additionalAttributes);
}

void FabricMaterial::createImageryLayerResolver(const omni::fabric::Path& path, uint64_t imageryLayerCount) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(path);

    FabricAttributesBuilder attributes;

    attributes.addAttribute(FabricTypes::inputs_imagery_layers_count, FabricTokens::inputs_imagery_layers_count);

    createAttributes(srw, path, attributes, FabricTokens::cesium_internal_imagery_layer_resolver);

    auto imageryLayerCountFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_imagery_layers_count);
    *imageryLayerCountFabric = static_cast<int>(imageryLayerCount);
}

void FabricMaterial::createFeatureIdIndex(const omni::fabric::Path& path) {
    createFeatureIdAttribute(path);
}

void FabricMaterial::createFeatureIdAttribute(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(path);

    FabricAttributesBuilder attributes;

    attributes.addAttribute(FabricTypes::inputs_primvar_name, FabricTokens::inputs_primvar_name);
    attributes.addAttribute(FabricTypes::inputs_null_feature_id, FabricTokens::inputs_null_feature_id);

    createAttributes(srw, path, attributes, FabricTokens::cesium_internal_feature_id_attribute_lookup);
}

void FabricMaterial::createFeatureIdTexture(const omni::fabric::Path& path) {
    static const auto additionalAttributes = std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>{{
        std::make_pair(FabricTypes::inputs_channels, FabricTokens::inputs_channels),
        std::make_pair(FabricTypes::inputs_channel_count, FabricTokens::inputs_channel_count),
        std::make_pair(FabricTypes::inputs_null_feature_id, FabricTokens::inputs_null_feature_id),
    }};

    return createTextureCommon(path, FabricTokens::cesium_internal_feature_id_texture_lookup, additionalAttributes);
}

void FabricMaterial::reset() {
    if (_usesDefaultMaterial) {
        setShaderValues(
            _shaderPath, GltfUtil::getDefaultMaterialInfo(), DEFAULT_DISPLAY_COLOR, DEFAULT_DISPLAY_OPACITY);
    }

    if (_materialDefinition.hasBaseColorTexture()) {
        setTextureValues(
            _baseColorTexturePath,
            _defaultTextureAssetPathToken,
            GltfUtil::getDefaultTextureInfo(),
            DEFAULT_TEXCOORD_INDEX);
    }

    for (const auto& featureIdIndexPath : _featureIdIndexPaths) {
        setFeatureIdIndexValues(featureIdIndexPath, DEFAULT_NULL_FEATURE_ID);
    }

    for (const auto& featureIdAttributePath : _featureIdAttributePaths) {
        setFeatureIdAttributeValues(featureIdAttributePath, DEFAULT_FEATURE_ID_PRIMVAR_NAME, DEFAULT_NULL_FEATURE_ID);
    }

    for (const auto& featureIdTexturePath : _featureIdTexturePaths) {
        setFeatureIdTextureValues(
            featureIdTexturePath,
            _defaultTransparentTextureAssetPathToken,
            GltfUtil::getDefaultTextureInfo(),
            DEFAULT_TEXCOORD_INDEX,
            DEFAULT_NULL_FEATURE_ID);
    }

    for (const auto& imageryLayerPath : _imageryLayerPaths) {
        setImageryLayerValues(
            imageryLayerPath,
            _defaultTransparentTextureAssetPathToken,
            GltfUtil::getDefaultTextureInfo(),
            DEFAULT_TEXCOORD_INDEX,
            DEFAULT_ALPHA);
    }

    for (const auto& path : _allPaths) {
        FabricUtil::setTilesetId(path, NO_TILESET_ID);
    }
}

void FabricMaterial::setMaterial(
    int64_t tilesetId,
    const MaterialInfo& materialInfo,
    const FeaturesInfo& featuresInfo,
    const std::shared_ptr<FabricTexture>& baseColorTexture,
    const std::vector<std::shared_ptr<FabricTexture>>& featureIdTextures,
    const glm::dvec3& displayColor,
    double displayOpacity,
    const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
    const std::vector<uint64_t>& featureIdIndexSetIndexMapping,
    const std::vector<uint64_t>& featureIdAttributeSetIndexMapping,
    const std::vector<uint64_t>& featureIdTextureSetIndexMapping) {

    if (stageDestroyed()) {
        return;
    }

    if (_usesDefaultMaterial) {
        _alphaMode = materialInfo.alphaMode;

        if (_debugRandomColors) {
            const auto r = glm::linearRand(0.0, 1.0);
            const auto g = glm::linearRand(0.0, 1.0);
            const auto b = glm::linearRand(0.0, 1.0);
            _debugColor = glm::dvec3(r, g, b);
        } else {
            _debugColor = DEFAULT_DEBUG_COLOR;
        }

        setShaderValues(_shaderPath, materialInfo, displayColor, displayOpacity);
    }

    if (_materialDefinition.hasBaseColorTexture()) {
        const auto& textureInfo = materialInfo.baseColorTexture.value();
        const auto& textureAssetPath = baseColorTexture->getAssetPathToken();
        const auto texcoordIndex = texcoordIndexMapping.at(textureInfo.setIndex);

        setTextureValues(_baseColorTexturePath, textureAssetPath, textureInfo, texcoordIndex);
    }

    const auto featureIdCounts = getFeatureIdCounts(_materialDefinition);

    for (uint64_t i = 0; i < featureIdCounts.indexCount; i++) {
        const auto featureIdSetIndex = featureIdIndexSetIndexMapping[i];
        const auto featureId = featuresInfo.featureIds[featureIdSetIndex];
        assert(std::holds_alternative<std::monostate>(featureId.featureIdStorage));
        const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
        const auto nullFeatureId = defaultValue(featureId.nullFeatureId, DEFAULT_NULL_FEATURE_ID);

        setFeatureIdIndexValues(featureIdPath, nullFeatureId);
    }

    for (uint64_t i = 0; i < featureIdCounts.attributeCount; i++) {
        const auto featureIdSetIndex = featureIdAttributeSetIndexMapping[i];
        const auto featureId = featuresInfo.featureIds[featureIdSetIndex];
        assert(std::holds_alternative<uint64_t>(featureId.featureIdStorage));
        const auto attributeSetIndex = std::get<uint64_t>(featureId.featureIdStorage);
        const auto attributeName = fmt::format("_FEATURE_ID_{}", attributeSetIndex);
        const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
        const auto nullFeatureId = defaultValue(featureId.nullFeatureId, DEFAULT_NULL_FEATURE_ID);

        setFeatureIdAttributeValues(featureIdPath, attributeName, nullFeatureId);
    }

    for (uint64_t i = 0; i < featureIdCounts.textureCount; i++) {
        const auto featureIdSetIndex = featureIdTextureSetIndexMapping[i];
        const auto& featureId = featuresInfo.featureIds[featureIdSetIndex];
        assert(std::holds_alternative<TextureInfo>(featureId.featureIdStorage));
        const auto& textureInfo = std::get<TextureInfo>(featureId.featureIdStorage);
        const auto& textureAssetPath = featureIdTextures[i]->getAssetPathToken();
        const auto texcoordIndex = texcoordIndexMapping.at(textureInfo.setIndex);
        const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
        const auto nullFeatureId = defaultValue(featureId.nullFeatureId, DEFAULT_NULL_FEATURE_ID);

        setFeatureIdTextureValues(featureIdPath, textureAssetPath, textureInfo, texcoordIndex, nullFeatureId);
    }

    for (const auto& path : _allPaths) {
        FabricUtil::setTilesetId(path, tilesetId);
    }
}

void FabricMaterial::createConnectionsToCopiedPaths() {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();
    const auto imageryLayerCount = getImageryLayerCount(_materialDefinition);
    const auto featureIdCount = getFeatureIdCounts(_materialDefinition).totalCount;

    for (const auto& copiedPath : _copiedBaseColorTexturePaths) {
        if (hasBaseColorTexture) {
            createConnection(srw, _baseColorTexturePath, copiedPath, FabricTokens::inputs_base_color_texture);
        }
    }

    for (const auto& copiedPath : _copiedImageryLayerPaths) {
        const auto indexFabric = srw.getAttributeRd<int>(copiedPath, FabricTokens::inputs_imagery_layer_index);
        const auto index = static_cast<uint64_t>(defaultValue(indexFabric, 0));

        if (index < imageryLayerCount) {
            createConnection(srw, _imageryLayerPaths[index], copiedPath, FabricTokens::inputs_imagery_layer);
        }
    }

    for (const auto& copiedPath : _copiedFeatureIdPaths) {
        const auto indexFabric = srw.getAttributeRd<int>(copiedPath, FabricTokens::inputs_feature_id_set_index);
        const auto index = static_cast<uint64_t>(defaultValue(indexFabric, 0));

        if (index < featureIdCount) {
            createConnection(srw, _featureIdPaths[index], copiedPath, FabricTokens::inputs_feature_id);
        }
    }
}

void FabricMaterial::destroyConnectionsToCopiedPaths() {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    for (const auto& copiedPath : _copiedBaseColorTexturePaths) {
        destroyConnection(srw, copiedPath, FabricTokens::inputs_base_color_texture);
    }

    for (const auto& copiedPath : _copiedImageryLayerPaths) {
        destroyConnection(srw, copiedPath, FabricTokens::inputs_imagery_layer);
    }

    for (const auto& copiedPath : _copiedFeatureIdPaths) {
        destroyConnection(srw, copiedPath, FabricTokens::inputs_feature_id);
    }
}

void FabricMaterial::setImageryLayer(
    const std::shared_ptr<FabricTexture>& texture,
    const TextureInfo& textureInfo,
    uint64_t imageryLayerIndex,
    double alpha,
    const std::unordered_map<uint64_t, uint64_t>& imageryTexcoordIndexMapping) {
    if (stageDestroyed()) {
        return;
    }

    if (imageryLayerIndex >= _imageryLayerPaths.size()) {
        return;
    }

    const auto& textureAssetPath = texture->getAssetPathToken();
    const auto texcoordIndex = imageryTexcoordIndexMapping.at(textureInfo.setIndex);
    const auto& imageryLayerPath = _imageryLayerPaths[imageryLayerIndex];
    setImageryLayerValues(imageryLayerPath, textureAssetPath, textureInfo, texcoordIndex, alpha);
}

void FabricMaterial::setImageryLayerAlpha(uint64_t imageryLayerIndex, double alpha) {
    if (stageDestroyed()) {
        return;
    }

    if (imageryLayerIndex >= _imageryLayerPaths.size()) {
        return;
    }

    const auto& imageryLayerPath = _imageryLayerPaths[imageryLayerIndex];
    setImageryLayerAlphaValue(imageryLayerPath, alpha);
}

void FabricMaterial::setDisplayColorAndOpacity(const glm::dvec3& displayColor, double displayOpacity) {
    if (stageDestroyed()) {
        return;
    }

    if (!_usesDefaultMaterial) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();

    auto tileColorFabric = srw.getAttributeWr<pxr::GfVec4f>(_shaderPath, FabricTokens::inputs_tile_color);
    auto alphaModeFabric = srw.getAttributeWr<int>(_shaderPath, FabricTokens::inputs_alpha_mode);

    *tileColorFabric = getTileColor(_debugColor, displayColor, displayOpacity);
    *alphaModeFabric = getAlphaMode(_alphaMode, displayOpacity);
}

void FabricMaterial::updateShaderInput(const omni::fabric::Path& path, const omni::fabric::Token& attributeName) {
    if (stageDestroyed()) {
        return;
    }

    const auto srw = UsdUtil::getFabricStageReaderWriter();
    const auto isrw = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();

    const auto copiedShaderPath = FabricUtil::getCopiedShaderPath(_materialPath, path);
    const auto attributesToCopy = std::vector<omni::fabric::TokenC>{attributeName};

    assert(isrw->primExists(srw.getId(), copiedShaderPath));

    isrw->copySpecifiedAttributes(
        srw.getId(), path, attributesToCopy.data(), copiedShaderPath, attributesToCopy.data(), attributesToCopy.size());

    if (attributeName == FabricTokens::inputs_imagery_layer_index ||
        attributeName == FabricTokens::inputs_feature_id_set_index) {
        destroyConnectionsToCopiedPaths();
        createConnectionsToCopiedPaths();
    }
}

void FabricMaterial::clearImageryLayer(uint64_t imageryLayerIndex) {
    if (stageDestroyed()) {
        return;
    }

    if (imageryLayerIndex >= _imageryLayerPaths.size()) {
        return;
    }

    const auto& imageryLayerPath = _imageryLayerPaths[imageryLayerIndex];
    setImageryLayerValues(
        imageryLayerPath,
        _defaultTransparentTextureAssetPathToken,
        GltfUtil::getDefaultTextureInfo(),
        DEFAULT_TEXCOORD_INDEX,
        DEFAULT_ALPHA);
}

void FabricMaterial::setShaderValues(
    const omni::fabric::Path& path,
    const MaterialInfo& materialInfo,
    const glm::dvec3& displayColor,
    double displayOpacity) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    auto tileColorFabric = srw.getAttributeWr<pxr::GfVec4f>(path, FabricTokens::inputs_tile_color);
    auto alphaCutoffFabric = srw.getAttributeWr<float>(path, FabricTokens::inputs_alpha_cutoff);
    auto alphaModeFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_alpha_mode);
    auto baseAlphaFabric = srw.getAttributeWr<float>(path, FabricTokens::inputs_base_alpha);
    auto baseColorFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(path, FabricTokens::inputs_base_color_factor);
    auto emissiveFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(path, FabricTokens::inputs_emissive_factor);
    auto metallicFactorFabric = srw.getAttributeWr<float>(path, FabricTokens::inputs_metallic_factor);
    auto roughnessFactorFabric = srw.getAttributeWr<float>(path, FabricTokens::inputs_roughness_factor);

    *tileColorFabric = getTileColor(_debugColor, displayColor, displayOpacity);
    *alphaCutoffFabric = static_cast<float>(materialInfo.alphaCutoff);
    *alphaModeFabric = getAlphaMode(_alphaMode, displayOpacity);
    *baseAlphaFabric = static_cast<float>(materialInfo.baseAlpha);
    *baseColorFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.baseColorFactor));
    *emissiveFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.emissiveFactor));
    *metallicFactorFabric = static_cast<float>(materialInfo.metallicFactor);
    *roughnessFactorFabric = static_cast<float>(materialInfo.roughnessFactor);
}

void FabricMaterial::setTextureValuesCommon(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const TextureInfo& textureInfo,
    uint64_t texcoordIndex) {

    auto srw = UsdUtil::getFabricStageReaderWriter();

    auto offset = textureInfo.offset;
    auto rotation = textureInfo.rotation;
    auto scale = textureInfo.scale;

    if (!textureInfo.flipVertical) {
        // gltf/pbr.mdl does texture transform math in glTF coordinates (top-left origin), so we needed to convert
        // the translation and scale parameters to work in that space. This doesn't handle rotation yet because we
        // haven't needed it for imagery layers.
        offset = {offset.x, 1.0 - offset.y - scale.y};
        scale = {scale.x, scale.y};
    }

    auto textureFabric = srw.getAttributeWr<omni::fabric::AssetPath>(path, FabricTokens::inputs_texture);
    auto texCoordIndexFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_tex_coord_index);
    auto wrapSFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_wrap_s);
    auto wrapTFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_wrap_t);
    auto offsetFabric = srw.getAttributeWr<pxr::GfVec2f>(path, FabricTokens::inputs_tex_coord_offset);
    auto rotationFabric = srw.getAttributeWr<float>(path, FabricTokens::inputs_tex_coord_rotation);
    auto scaleFabric = srw.getAttributeWr<pxr::GfVec2f>(path, FabricTokens::inputs_tex_coord_scale);

    textureFabric->assetPath = textureAssetPathToken;
    textureFabric->resolvedPath = pxr::TfToken();
    *texCoordIndexFabric = static_cast<int>(texcoordIndex);
    *wrapSFabric = textureInfo.wrapS;
    *wrapTFabric = textureInfo.wrapT;
    *offsetFabric = UsdUtil::glmToUsdVector(glm::fvec2(offset));
    *rotationFabric = static_cast<float>(rotation);
    *scaleFabric = UsdUtil::glmToUsdVector(glm::fvec2(scale));
}

void FabricMaterial::setTextureValues(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const TextureInfo& textureInfo,
    uint64_t texcoordIndex) {
    setTextureValuesCommon(path, textureAssetPathToken, textureInfo, texcoordIndex);
}

void FabricMaterial::setImageryLayerValues(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const TextureInfo& textureInfo,
    uint64_t texcoordIndex,
    double alpha) {
    setTextureValuesCommon(path, textureAssetPathToken, textureInfo, texcoordIndex);
    setImageryLayerAlphaValue(path, alpha);
}

void FabricMaterial::setImageryLayerAlphaValue(const omni::fabric::Path& path, double alpha) {
    auto srw = UsdUtil::getFabricStageReaderWriter();
    auto alphaFabric = srw.getAttributeWr<float>(path, FabricTokens::inputs_alpha);
    *alphaFabric = static_cast<float>(alpha);
}

void FabricMaterial::setFeatureIdIndexValues(const omni::fabric::Path& path, int nullFeatureId) {
    setFeatureIdAttributeValues(path, pxr::UsdTokens->vertexId.GetString(), nullFeatureId);
}

void FabricMaterial::setFeatureIdAttributeValues(
    const omni::fabric::Path& path,
    const std::string& primvarName,
    int nullFeatureId) {

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto primvarNameSize = primvarName.size();
    srw.setArrayAttributeSize(path, FabricTokens::inputs_primvar_name, primvarNameSize);
    auto primvarNameFabric = srw.getArrayAttributeWr<uint8_t>(path, FabricTokens::inputs_primvar_name);
    memcpy(primvarNameFabric.data(), primvarName.data(), primvarNameSize);

    auto nullFeatureIdFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_null_feature_id);
    *nullFeatureIdFabric = nullFeatureId;
}

void FabricMaterial::setFeatureIdTextureValues(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const TextureInfo& textureInfo,
    uint64_t texcoordIndex,
    int nullFeatureId) {

    setTextureValuesCommon(path, textureAssetPathToken, textureInfo, texcoordIndex);

    auto channelCount = glm::min(textureInfo.channels.size(), uint64_t(4));
    auto channels = glm::u8vec4(0);
    for (uint64_t i = 0; i < channelCount; i++) {
        channels[i] = textureInfo.channels[i];
    }
    channelCount = glm::max(channelCount, uint64_t(1));

    auto srw = UsdUtil::getFabricStageReaderWriter();
    auto channelCountFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_channel_count);
    auto channelsFabric = srw.getAttributeWr<glm::i32vec4>(path, FabricTokens::inputs_channels);
    auto nullFeatureIdFabric = srw.getAttributeWr<int>(path, FabricTokens::inputs_null_feature_id);

    *channelCountFabric = static_cast<int>(channelCount);
    *channelsFabric = static_cast<glm::i32vec4>(channels);
    *nullFeatureIdFabric = nullFeatureId;
}

bool FabricMaterial::stageDestroyed() {
    // Add this guard to all public member functions, including constructors and destructors. Tile render resources can
    // continue to be processed asynchronously even after the tileset and USD stage have been destroyed, so prevent any
    // operations that would modify the stage.
    return _stageId != UsdUtil::getUsdStageId();
}

} // namespace cesium::omniverse
