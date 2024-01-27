#include "cesium/omniverse/FabricMaterial.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricFeaturesInfo.h"
#include "cesium/omniverse/FabricImageryLayersInfo.h"
#include "cesium/omniverse/FabricMaterialDescriptor.h"
#include "cesium/omniverse/FabricMaterialInfo.h"
#include "cesium/omniverse/FabricPropertyDescriptor.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricTexture.h"
#include "cesium/omniverse/FabricTextureInfo.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Logger.h"
#include "cesium/omniverse/MetadataUtil.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumGltf/MeshPrimitive.h>
#include <glm/gtc/random.hpp>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/SimStageWithHistory.h>
#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

namespace {

// Should match imagery_layers length in cesium.mdl
const uint64_t MAX_RASTER_OVERLAY_LAYERS_COUNT = 16;

const auto DEFAULT_DEBUG_COLOR = glm::dvec3(1.0, 1.0, 1.0);
const auto DEFAULT_ALPHA = 1.0f;
const auto DEFAULT_DISPLAY_COLOR = glm::dvec3(1.0, 1.0, 1.0);
const auto DEFAULT_DISPLAY_OPACITY = 1.0;
const auto DEFAULT_TEXCOORD_INDEX = uint64_t(0);
const auto DEFAULT_FEATURE_ID_PRIMVAR_NAME = std::string("_FEATURE_ID_0");
const auto DEFAULT_NULL_FEATURE_ID = -1;
const auto DEFAULT_OFFSET = 0;
const auto DEFAULT_SCALE = 1;
const auto DEFAULT_NO_DATA = 0;
const auto DEFAULT_VALUE = 0;

struct FeatureIdCounts {
    uint64_t indexCount;
    uint64_t attributeCount;
    uint64_t textureCount;
    uint64_t totalCount;
};

FeatureIdCounts getFeatureIdCounts(const FabricMaterialDescriptor& materialDescriptor) {
    const auto& featureIdTypes = materialDescriptor.getFeatureIdTypes();
    auto featureIdCount = featureIdTypes.size();

    uint64_t indexCount = 0;
    uint64_t attributeCount = 0;
    uint64_t textureCount = 0;

    for (uint64_t i = 0; i < featureIdCount; ++i) {
        const auto featureIdType = featureIdTypes[i];
        switch (featureIdType) {
            case FabricFeatureIdType::INDEX:
                ++indexCount;
                break;
            case FabricFeatureIdType::ATTRIBUTE:
                ++attributeCount;
                break;
            case FabricFeatureIdType::TEXTURE:
                ++textureCount;
                break;
        }
    }

    return FeatureIdCounts{indexCount, attributeCount, textureCount, featureIdCount};
}

struct RasterOverlayLayerIndices {
    std::vector<uint64_t> overlayRasterOverlayLayerIndices;
    std::vector<uint64_t> clippingRasterOverlayLayerIndices;
};

RasterOverlayLayerIndices getRasterOverlayLayerIndices(const Context& context, const FabricMaterialDescriptor& materialDescriptor) {
    uint64_t overlayRasterOverlayLayerCount = 0;
    uint64_t clippingRasterOverlayLayerCount = 0;
    uint64_t totalRasterOverlayLayerCount = 0;

    std::vector<uint64_t> overlayRasterOverlayLayerIndices;
    std::vector<uint64_t> clippingRasterOverlayLayerIndices;

    for (const auto& method : materialDescriptor.getRasterOverlayRenderMethods()) {
        switch (method) {
            case FabricOverlayRenderMethod::OVERLAY:
                if (overlayRasterOverlayLayerCount < MAX_RASTER_OVERLAY_LAYERS_COUNT) {
                    overlayRasterOverlayLayerIndices.push_back(totalRasterOverlayLayerCount);
                }
                ++overlayRasterOverlayLayerCount;
                break;
            case FabricOverlayRenderMethod::CLIPPING:
                if (clippingRasterOverlayLayerCount < MAX_RASTER_OVERLAY_LAYERS_COUNT) {
                    clippingRasterOverlayLayerIndices.push_back(totalRasterOverlayLayerCount);
                }
                ++clippingRasterOverlayLayerCount;
                break;
        }
        ++totalRasterOverlayLayerCount;
    }

    if (overlayRasterOverlayLayerCount > MAX_RASTER_OVERLAY_LAYERS_COUNT) {
        context.getLogger()->warn(
            "Number of overlay imagery layers ({}) exceeds maximum imagery layer count ({}). Excess imagery layers "
            "will be ignored.",
            overlayRasterOverlayLayerCount,
            MAX_RASTER_OVERLAY_LAYERS_COUNT);
    }

    if (clippingRasterOverlayLayerCount > MAX_RASTER_OVERLAY_LAYERS_COUNT) {
        context.getLogger()->warn(
            "Number of clipping imagery layers ({}) exceeds maximum imagery layer count ({}). Excess imagery layers "
            "will be ignored.",
            clippingRasterOverlayLayerCount,
            MAX_RASTER_OVERLAY_LAYERS_COUNT);
    }

    return RasterOverlayLayerIndices{std::move(overlayRasterOverlayLayerIndices), std::move(clippingRasterOverlayLayerIndices)};
}

uint64_t getRasterOverlayLayerCount(const FabricMaterialDescriptor& materialDescriptor) {
    return materialDescriptor.getRasterOverlayRenderMethods().size();
}

bool isClippingEnabled(const FabricMaterialDescriptor& materialDescriptor) {
    return CppUtil::contains(materialDescriptor.getRasterOverlayRenderMethods(), FabricOverlayRenderMethod::CLIPPING);
}

FabricAlphaMode
getInitialAlphaMode(const FabricMaterialDescriptor& materialDescriptor, const FabricMaterialInfo& materialInfo) {
    if (materialInfo.alphaMode == FabricAlphaMode::BLEND) {
        return materialInfo.alphaMode;
    }

    if (isClippingEnabled(materialDescriptor)) {
        return FabricAlphaMode::MASK;
    }

    return materialInfo.alphaMode;
}

int getAlphaMode(FabricAlphaMode alphaMode, double displayOpacity) {
    return static_cast<int>(displayOpacity < 1.0 ? FabricAlphaMode::BLEND : alphaMode);
}

glm::dvec4 getTileColor(const glm::dvec3& debugColor, const glm::dvec3& displayColor, double displayOpacity) {
    return {debugColor * displayColor, displayOpacity};
}

void createConnection(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& outputPath,
    const omni::fabric::Path& inputPath,
    const omni::fabric::Token& inputName) {
    fabricStage.createConnection(inputPath, inputName, omni::fabric::Connection{outputPath, FabricTokens::outputs_out});
}

void destroyConnection(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& inputPath,
    const omni::fabric::Token& inputName) {
    fabricStage.destroyConnection(inputPath, inputName);
}

template <DataType T>
constexpr DataTypeUtil::GetMdlInternalPropertyTransformedType<DataTypeUtil::getMdlInternalPropertyType<T>()>
getOffset(const FabricPropertyInfo<T>& info) {
    constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<T>();
    using TransformedType = DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<mdlType>;
    return static_cast<MdlTransformedType>(info.offset.value_or(TransformedType{DEFAULT_OFFSET}));
}

template <DataType T>
constexpr DataTypeUtil::GetMdlInternalPropertyTransformedType<DataTypeUtil::getMdlInternalPropertyType<T>()>
getScale(const FabricPropertyInfo<T>& info) {
    constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<T>();
    using TransformedType = DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<mdlType>;
    return static_cast<MdlTransformedType>(info.scale.value_or(TransformedType{DEFAULT_SCALE}));
}

template <DataType T>
constexpr DataTypeUtil::GetMdlInternalPropertyRawType<DataTypeUtil::getMdlInternalPropertyType<T>()>
getNoData(const FabricPropertyInfo<T>& info) {
    constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<T>();
    using RawType = DataTypeUtil::GetNativeType<T>;
    using MdlRawType = DataTypeUtil::GetMdlInternalPropertyRawType<mdlType>;
    return static_cast<MdlRawType>(info.noData.value_or(RawType{DEFAULT_NO_DATA}));
}

template <DataType T>
constexpr DataTypeUtil::GetMdlInternalPropertyTransformedType<DataTypeUtil::getMdlInternalPropertyType<T>()>
getDefaultValue(const FabricPropertyInfo<T>& info) {
    constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<T>();
    using TransformedType = DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<mdlType>;
    return static_cast<MdlTransformedType>(info.defaultValue.value_or(TransformedType{DEFAULT_VALUE}));
}

template <DataType T>
constexpr DataTypeUtil::GetMdlInternalPropertyRawType<DataTypeUtil::getMdlInternalPropertyType<T>()> getMaximumValue() {
    constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<T>();
    using RawComponentType = DataTypeUtil::GetNativeType<DataTypeUtil::getComponentType<T>()>;
    using MdlRawType = DataTypeUtil::GetMdlInternalPropertyRawType<mdlType>;

    if constexpr (DataTypeUtil::isNormalized<T>()) {
        return MdlRawType{std::numeric_limits<RawComponentType>::max()};
    }

    return MdlRawType{0};
}

void createAttributes(
    const Context& context,
    omni::fabric::StageReaderWriter& fabricStage,
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
    const auto inputsExcludeFromWhiteModeFabric = fabricStage.getAttributeWr<bool>(path, FabricTokens::inputs_excludeFromWhiteMode);
    const auto infoImplementationSourceFabric = fabricStage.getAttributeWr<omni::fabric::TokenC>(path, FabricTokens::info_implementationSource);
    const auto infoMdlSourceAssetFabric = fabricStage.getAttributeWr<omni::fabric::AssetPath>(path, FabricTokens::info_mdl_sourceAsset);
    const auto infoMdlSourceAssetSubIdentifierFabric = fabricStage.getAttributeWr<omni::fabric::TokenC>(path, FabricTokens::info_mdl_sourceAsset_subIdentifier);
    // clang-format on

    fabricStage.setArrayAttributeSize(path, FabricTokens::_paramColorSpace, 0);
    fabricStage.setArrayAttributeSize(path, FabricTokens::_sdrMetadata, 0);

    *inputsExcludeFromWhiteModeFabric = false;
    *infoImplementationSourceFabric = FabricTokens::sourceAsset;
    infoMdlSourceAssetFabric->assetPath = context.getCesiumMdlPathToken();
    infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
    *infoMdlSourceAssetSubIdentifierFabric = subidentifier;
}

void setTextureValuesCommon(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const FabricTextureInfo& textureInfo,
    uint64_t texcoordIndex) {

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

    const auto textureFabric = fabricStage.getAttributeWr<omni::fabric::AssetPath>(path, FabricTokens::inputs_texture);
    const auto texCoordIndexFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_tex_coord_index);
    const auto wrapSFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_wrap_s);
    const auto wrapTFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_wrap_t);
    const auto offsetFabric = fabricStage.getAttributeWr<pxr::GfVec2f>(path, FabricTokens::inputs_tex_coord_offset);
    const auto rotationFabric = fabricStage.getAttributeWr<float>(path, FabricTokens::inputs_tex_coord_rotation);
    const auto scaleFabric = fabricStage.getAttributeWr<pxr::GfVec2f>(path, FabricTokens::inputs_tex_coord_scale);

    textureFabric->assetPath = textureAssetPathToken;
    textureFabric->resolvedPath = pxr::TfToken();
    *texCoordIndexFabric = static_cast<int>(texcoordIndex);
    *wrapSFabric = textureInfo.wrapS;
    *wrapTFabric = textureInfo.wrapT;
    *offsetFabric = UsdUtil::glmToUsdVector(glm::fvec2(offset));
    *rotationFabric = static_cast<float>(rotation);
    *scaleFabric = UsdUtil::glmToUsdVector(glm::fvec2(scale));
}

void setTextureValuesCommonChannels(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const FabricTextureInfo& textureInfo,
    uint64_t texcoordIndex) {

    setTextureValuesCommon(fabricStage, path, textureAssetPathToken, textureInfo, texcoordIndex);

    auto channelCount = glm::min(textureInfo.channels.size(), uint64_t(4));
    auto channels = glm::u8vec4(0);
    for (uint64_t i = 0; i < channelCount; ++i) {
        channels[i] = textureInfo.channels[i];
    }
    channelCount = glm::max(channelCount, uint64_t(1));

    const auto channelsFabric = fabricStage.getAttributeWr<glm::i32vec4>(path, FabricTokens::inputs_channels);
    *channelsFabric = static_cast<glm::i32vec4>(channels);

    if (fabricStage.attributeExists(path, FabricTokens::inputs_channel_count)) {
        const auto channelCountFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_channel_count);
        *channelCountFabric = static_cast<int>(channelCount);
    }
}

std::string getStringFabric(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    omni::fabric::TokenC attributeName) {
    const auto valueFabric = fabricStage.getArrayAttributeRd<uint8_t>(path, attributeName);
    return {reinterpret_cast<const char*>(valueFabric.data()), valueFabric.size()};
}

void setStringFabric(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    omni::fabric::TokenC attributeName,
    const std::string& value) {
    fabricStage.setArrayAttributeSize(path, attributeName, value.size());
    const auto valueFabric = fabricStage.getArrayAttributeWr<uint8_t>(path, attributeName);
    memcpy(valueFabric.data(), value.data(), value.size());
}

template <MdlInternalPropertyType T>
void setPropertyValues(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& offset,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& scale,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& maximumValue,
    bool hasNoData,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& noData,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& defaultValue) {

    using MdlRawType = DataTypeUtil::GetMdlInternalPropertyRawType<T>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<T>;

    const auto hasNoDataFabric = fabricStage.getAttributeWr<bool>(path, FabricTokens::inputs_has_no_data);
    const auto noDataFabric = fabricStage.getAttributeWr<MdlRawType>(path, FabricTokens::inputs_no_data);
    const auto defaultValueFabric =
        fabricStage.getAttributeWr<MdlTransformedType>(path, FabricTokens::inputs_default_value);

    *hasNoDataFabric = hasNoData;
    *noDataFabric = static_cast<MdlRawType>(noData);
    *defaultValueFabric = static_cast<MdlTransformedType>(defaultValue);

    if (fabricStage.attributeExists(path, FabricTokens::inputs_offset)) {
        const auto offsetFabric = fabricStage.getAttributeWr<MdlTransformedType>(path, FabricTokens::inputs_offset);
        *offsetFabric = static_cast<MdlTransformedType>(offset);
    }

    if (fabricStage.attributeExists(path, FabricTokens::inputs_scale)) {
        const auto scaleFabric = fabricStage.getAttributeWr<MdlTransformedType>(path, FabricTokens::inputs_scale);
        *scaleFabric = static_cast<MdlTransformedType>(scale);
    }

    if (fabricStage.attributeExists(path, FabricTokens::inputs_maximum_value)) {
        const auto maximumValueFabric =
            fabricStage.getAttributeWr<MdlRawType>(path, FabricTokens::inputs_maximum_value);
        *maximumValueFabric = static_cast<MdlRawType>(maximumValue);
    }
}

template <MdlInternalPropertyType T>
void setPropertyAttributePropertyValues(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const std::string& primvarName,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& offset,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& scale,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& maximumValue,
    bool hasNoData,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& noData,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& defaultValue) {

    setStringFabric(fabricStage, path, FabricTokens::inputs_primvar_name, primvarName);
    setPropertyValues<T>(fabricStage, path, offset, scale, maximumValue, hasNoData, noData, defaultValue);
}

template <MdlInternalPropertyType T>
void setPropertyTexturePropertyValues(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const FabricTextureInfo& textureInfo,
    uint64_t texcoordIndex,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& offset,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& scale,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& maximumValue,
    bool hasNoData,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& noData,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& defaultValue) {

    setTextureValuesCommonChannels(fabricStage, path, textureAssetPathToken, textureInfo, texcoordIndex);
    setPropertyValues<T>(fabricStage, path, offset, scale, maximumValue, hasNoData, noData, defaultValue);
}

template <MdlInternalPropertyType T>
void setPropertyTablePropertyValues(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const pxr::TfToken& propertyTableTextureAssetPathToken,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& offset,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& scale,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& maximumValue,
    bool hasNoData,
    const DataTypeUtil::GetMdlInternalPropertyRawType<T>& noData,
    const DataTypeUtil::GetMdlInternalPropertyTransformedType<T>& defaultValue) {

    const auto textureFabric =
        fabricStage.getAttributeWr<omni::fabric::AssetPath>(path, FabricTokens::inputs_property_table_texture);
    textureFabric->assetPath = propertyTableTextureAssetPathToken;
    textureFabric->resolvedPath = pxr::TfToken();

    setPropertyValues<T>(fabricStage, path, offset, scale, maximumValue, hasNoData, noData, defaultValue);
}

template <MdlInternalPropertyType T>
void clearPropertyAttributeProperty(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    using MdlRawType = DataTypeUtil::GetMdlInternalPropertyRawType<T>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<T>;

    setPropertyAttributePropertyValues<T>(
        fabricStage,
        path,
        "",
        MdlTransformedType{0},
        MdlTransformedType{0},
        MdlRawType{0},
        false,
        MdlRawType{0},
        MdlTransformedType{0});
}

template <MdlInternalPropertyType T>
void clearPropertyTextureProperty(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const pxr::TfToken& defaultTransparentTextureAssetPathToken) {
    using MdlRawType = DataTypeUtil::GetMdlInternalPropertyRawType<T>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<T>;

    setPropertyTexturePropertyValues<T>(
        fabricStage,
        path,
        defaultTransparentTextureAssetPathToken,
        GltfUtil::getDefaultTextureInfo(),
        DEFAULT_TEXCOORD_INDEX,
        MdlTransformedType{0},
        MdlTransformedType{0},
        MdlRawType{0},
        false,
        MdlRawType{0},
        MdlTransformedType{0});
}

template <MdlInternalPropertyType T>
void clearPropertyTableProperty(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const pxr::TfToken& defaultTransparentTextureAssetPathToken) {
    using MdlRawType = DataTypeUtil::GetMdlInternalPropertyRawType<T>;
    using MdlTransformedType = DataTypeUtil::GetMdlInternalPropertyTransformedType<T>;

    setPropertyTablePropertyValues<T>(
        fabricStage,
        path,
        defaultTransparentTextureAssetPathToken,
        MdlTransformedType{0},
        MdlTransformedType{0},
        MdlRawType{0},
        false,
        MdlRawType{0},
        MdlTransformedType{0});
}

} // namespace

FabricMaterial::FabricMaterial(
    Context* pContext,
    const omni::fabric::Path& path,
    const FabricMaterialDescriptor& materialDescriptor,
    const pxr::TfToken& defaultWhiteTextureAssetPathToken,
    const pxr::TfToken& defaultTransparentTextureAssetPathToken,
    bool debugRandomColors,
    int64_t poolId)
    : _pContext(pContext)
    , _materialPath(path)
    , _materialDescriptor(materialDescriptor)
    , _defaultWhiteTextureAssetPathToken(defaultWhiteTextureAssetPathToken)
    , _defaultTransparentTextureAssetPathToken(defaultTransparentTextureAssetPathToken)
    , _debugRandomColors(debugRandomColors)
    , _poolId(poolId)
    , _stageId(pContext->getUsdStageId())
    , _usesDefaultMaterial(!materialDescriptor.hasTilesetMaterial()) {

    if (stageDestroyed()) {
        return;
    }

    initializeNodes();

    if (_usesDefaultMaterial) {
        initializeDefaultMaterial();
    } else {
        initializeExistingMaterial(FabricUtil::toFabricPath(materialDescriptor.getTilesetMaterialPath()));
    }

    reset();
}

FabricMaterial::~FabricMaterial() {
    if (stageDestroyed()) {
        return;
    }

    for (const auto& path : _allPaths) {
        FabricUtil::destroyPrim(_pContext->getFabricStage(), path);
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

const FabricMaterialDescriptor& FabricMaterial::getMaterialDescriptor() const {
    return _materialDescriptor;
}

int64_t FabricMaterial::getPoolId() const {
    return _poolId;
}

void FabricMaterial::initializeNodes() {
    auto& fabricStage = _pContext->getFabricStage();

    // Create base color texture
    const auto hasBaseColorTexture = _materialDescriptor.hasBaseColorTexture();
    if (hasBaseColorTexture) {
        const auto baseColorTexturePath = FabricUtil::joinPaths(_materialPath, FabricTokens::base_color_texture);
        createTexture(baseColorTexturePath);
        _baseColorTexturePath = baseColorTexturePath;
        _allPaths.push_back(baseColorTexturePath);
    }

    // Create imagery layers
    const auto rasterOverlayLayerCount = getRasterOverlayLayerCount(_materialDescriptor);
    _rasterOverlayLayerPaths.reserve(rasterOverlayLayerCount);
    for (uint64_t i = 0; i < rasterOverlayLayerCount; ++i) {
        const auto rasterOverlayLayerPath = FabricUtil::joinPaths(_materialPath, FabricTokens::imagery_layer_n(i));
        createRasterOverlayLayer(rasterOverlayLayerPath);
        _rasterOverlayLayerPaths.push_back(rasterOverlayLayerPath);
        _allPaths.push_back(rasterOverlayLayerPath);
    }

    // Create feature ids
    const auto& featureIdTypes = _materialDescriptor.getFeatureIdTypes();
    const auto featureIdCounts = getFeatureIdCounts(_materialDescriptor);
    _featureIdPaths.reserve(featureIdCounts.totalCount);
    _featureIdIndexPaths.reserve(featureIdCounts.indexCount);
    _featureIdAttributePaths.reserve(featureIdCounts.attributeCount);
    _featureIdTexturePaths.reserve(featureIdCounts.textureCount);

    for (uint64_t i = 0; i < featureIdCounts.totalCount; ++i) {
        const auto featureIdType = featureIdTypes[i];
        const auto featureIdPath = FabricUtil::joinPaths(_materialPath, FabricTokens::feature_id_n(i));
        switch (featureIdType) {
            case FabricFeatureIdType::INDEX:
                createFeatureIdIndex(featureIdPath);
                _featureIdIndexPaths.push_back(featureIdPath);
                break;
            case FabricFeatureIdType::ATTRIBUTE:
                createFeatureIdAttribute(featureIdPath);
                _featureIdAttributePaths.push_back(featureIdPath);
                break;
            case FabricFeatureIdType::TEXTURE:
                createFeatureIdTexture(featureIdPath);
                _featureIdTexturePaths.push_back(featureIdPath);
                break;
        }
        _featureIdPaths.push_back(featureIdPath);
        _allPaths.push_back(featureIdPath);
    }

    // Create properties
    const auto& properties = _materialDescriptor.getStyleableProperties();
    for (uint64_t i = 0; i < properties.size(); ++i) {
        const auto& property = properties[i];
        const auto storageType = property.storageType;
        const auto type = property.type;
        const auto& propertyPath = FabricUtil::joinPaths(_materialPath, FabricTokens::property_n(i));
        switch (storageType) {
            case FabricPropertyStorageType::ATTRIBUTE:
                createPropertyAttributeProperty(propertyPath, type);
                _propertyAttributePropertyPaths[type].push_back(propertyPath);
                break;
            case FabricPropertyStorageType::TEXTURE:
                createPropertyTextureProperty(propertyPath, type);
                _propertyTexturePropertyPaths[type].push_back(propertyPath);
                break;
            case FabricPropertyStorageType::TABLE:
                createPropertyTableProperty(propertyPath, type);
                _propertyTablePropertyPaths[type].push_back(propertyPath);
                // Create connection from the feature id node to the property table property node
                const auto featureIdSetIndex = property.featureIdSetIndex;
                const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
                createConnection(fabricStage, featureIdPath, propertyPath, FabricTokens::inputs_feature_id);
                break;
        }

        _propertyPaths.push_back(propertyPath);
        _allPaths.push_back(propertyPath);
    }
}

void FabricMaterial::initializeDefaultMaterial() {
    auto& fabricStage = _pContext->getFabricStage();

    const auto rasterOverlayLayerIndices = getRasterOverlayLayerIndices(*_pContext, _materialDescriptor);
    const auto hasBaseColorTexture = _materialDescriptor.hasBaseColorTexture();

    // Create material
    const auto& materialPath = _materialPath;
    createMaterial(materialPath);
    _allPaths.push_back(materialPath);

    // Create shader
    const auto shaderPath = FabricUtil::joinPaths(materialPath, FabricTokens::cesium_internal_material);
    createShader(shaderPath);
    _shaderPath = shaderPath;
    _allPaths.push_back(shaderPath);

    const auto& overlayRasterOverlayLayerIndices = rasterOverlayLayerIndices.overlayRasterOverlayLayerIndices;
    const auto& clippingRasterOverlayLayerIndices = rasterOverlayLayerIndices.clippingRasterOverlayLayerIndices;
    const auto overlayRasterOverlayLayerCount = overlayRasterOverlayLayerIndices.size();
    const auto clippingRasterOverlayLayerCount = clippingRasterOverlayLayerIndices.size();

    // Create overlay imagery layer resolver if there are multiple overlay imagery layers
    if (overlayRasterOverlayLayerCount > 1) {
        const auto rasterOverlayLayerResolverPath = FabricUtil::joinPaths(materialPath, FabricTokens::imagery_layer_resolver);
        createRasterOverlayLayerResolver(rasterOverlayLayerResolverPath, overlayRasterOverlayLayerCount);
        _overlayRasterOverlayLayerResolverPath = rasterOverlayLayerResolverPath;
        _allPaths.push_back(rasterOverlayLayerResolverPath);
    }

    // Create clipping imagery layer resolver if there are multiple clipping imagery layers
    if (clippingRasterOverlayLayerCount > 1) {
        const auto clippingRasterOverlayLayerResolverPath =
            FabricUtil::joinPaths(materialPath, FabricTokens::clipping_imagery_layer_resolver);
        createClippingRasterOverlayLayerResolver(clippingRasterOverlayLayerResolverPath, clippingRasterOverlayLayerCount);
        _clippingRasterOverlayLayerResolverPath = clippingRasterOverlayLayerResolverPath;
        _allPaths.push_back(_clippingRasterOverlayLayerResolverPath);
    }

    // Create connection from shader to material
    createConnection(fabricStage, shaderPath, materialPath, FabricTokens::outputs_mdl_surface);
    createConnection(fabricStage, shaderPath, materialPath, FabricTokens::outputs_mdl_displacement);
    createConnection(fabricStage, shaderPath, materialPath, FabricTokens::outputs_mdl_volume);

    // Create connection from base color texture to shader
    if (hasBaseColorTexture) {
        createConnection(fabricStage, _baseColorTexturePath, shaderPath, FabricTokens::inputs_base_color_texture);
    }

    if (overlayRasterOverlayLayerCount == 1) {
        // Create connection from imagery layer to shader
        const auto& rasterOverlayLayerPath = _rasterOverlayLayerPaths[overlayRasterOverlayLayerIndices.front()];
        createConnection(fabricStage, rasterOverlayLayerPath, shaderPath, FabricTokens::inputs_imagery_layer);
    } else if (overlayRasterOverlayLayerCount > 1) {
        // Create connection from imagery layer resolver to shader
        createConnection(fabricStage, _overlayRasterOverlayLayerResolverPath, shaderPath, FabricTokens::inputs_imagery_layer);

        // Create connections from imagery layers to imagery layer resolver
        for (uint64_t i = 0; i < overlayRasterOverlayLayerCount; ++i) {
            const auto& rasterOverlayLayerPath = _rasterOverlayLayerPaths[overlayRasterOverlayLayerIndices[i]];
            createConnection(
                fabricStage,
                rasterOverlayLayerPath,
                _overlayRasterOverlayLayerResolverPath,
                FabricTokens::inputs_imagery_layer_n(i));
        }
    }

    if (clippingRasterOverlayLayerCount == 1) {
        // Create connection from imagery layer to shader
        const auto& rasterOverlayLayerPath = _rasterOverlayLayerPaths[clippingRasterOverlayLayerIndices.front()];
        createConnection(fabricStage, rasterOverlayLayerPath, shaderPath, FabricTokens::inputs_alpha_clip);
    } else if (clippingRasterOverlayLayerCount > 1) {
        // Create connection from imagery layer resolver to shader
        createConnection(fabricStage, _clippingRasterOverlayLayerResolverPath, shaderPath, FabricTokens::inputs_alpha_clip);

        // Create connections from imagery layers to imagery layer resolver
        for (uint64_t i = 0; i < clippingRasterOverlayLayerCount; ++i) {
            const auto& rasterOverlayLayerPath = _rasterOverlayLayerPaths[clippingRasterOverlayLayerIndices[i]];
            createConnection(
                fabricStage,
                rasterOverlayLayerPath,
                _clippingRasterOverlayLayerResolverPath,
                FabricTokens::inputs_imagery_layer_n(i));
        }
    }
}

void FabricMaterial::initializeExistingMaterial(const omni::fabric::Path& path) {
    auto& fabricStage = _pContext->getFabricStage();

    const auto copiedPaths = FabricUtil::copyMaterial(fabricStage, path, _materialPath);

    for (const auto& copiedPath : copiedPaths) {
        fabricStage.createAttribute(copiedPath, FabricTokens::_cesium_tilesetId, FabricTypes::_cesium_tilesetId);
        _allPaths.push_back(copiedPath);

        const auto mdlIdentifier = FabricUtil::getMdlIdentifier(fabricStage, copiedPath);

        if (mdlIdentifier == FabricTokens::cesium_base_color_texture_float4) {
            _copiedBaseColorTexturePaths.push_back(copiedPath);
        } else if (mdlIdentifier == FabricTokens::cesium_imagery_layer_float4) {
            _copiedRasterOverlayLayerPaths.push_back(copiedPath);
        } else if (mdlIdentifier == FabricTokens::cesium_feature_id_int) {
            _copiedFeatureIdPaths.push_back(copiedPath);
        } else if (FabricUtil::isCesiumPropertyNode(mdlIdentifier)) {
            _copiedPropertyPaths.push_back(copiedPath);
        }
    }

    createConnectionsToCopiedPaths();
    createConnectionsToProperties();
}

void FabricMaterial::createMaterial(const omni::fabric::Path& path) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);

    FabricAttributesBuilder attributes(_pContext);

    attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);

    attributes.createAttributes(path);
}

void FabricMaterial::createShader(const omni::fabric::Path& path) {
    auto& fabricStage = _pContext->getFabricStage();

    fabricStage.createPrim(path);

    FabricAttributesBuilder attributes(_pContext);

    attributes.addAttribute(FabricTypes::inputs_tile_color, FabricTokens::inputs_tile_color);
    attributes.addAttribute(FabricTypes::inputs_alpha_cutoff, FabricTokens::inputs_alpha_cutoff);
    attributes.addAttribute(FabricTypes::inputs_alpha_mode, FabricTokens::inputs_alpha_mode);
    attributes.addAttribute(FabricTypes::inputs_base_alpha, FabricTokens::inputs_base_alpha);
    attributes.addAttribute(FabricTypes::inputs_base_color_factor, FabricTokens::inputs_base_color_factor);
    attributes.addAttribute(FabricTypes::inputs_emissive_factor, FabricTokens::inputs_emissive_factor);
    attributes.addAttribute(FabricTypes::inputs_metallic_factor, FabricTokens::inputs_metallic_factor);
    attributes.addAttribute(FabricTypes::inputs_roughness_factor, FabricTokens::inputs_roughness_factor);

    createAttributes(*_pContext, fabricStage, path, attributes, FabricTokens::cesium_internal_material);
}

void FabricMaterial::createTextureCommon(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subIdentifier,
    const std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>& additionalAttributes) {
    auto& fabricStage = _pContext->getFabricStage();

    fabricStage.createPrim(path);

    FabricAttributesBuilder attributes(_pContext);

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

    createAttributes(*_pContext, fabricStage, path, attributes, subIdentifier);

    // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
    fabricStage.setArrayAttributeSize(path, FabricTokens::_paramColorSpace, 2);
    const auto paramColorSpaceFabric =
        fabricStage.getArrayAttributeWr<omni::fabric::TokenC>(path, FabricTokens::_paramColorSpace);
    paramColorSpaceFabric[0] = FabricTokens::inputs_texture;
    paramColorSpaceFabric[1] = FabricTokens::_auto;
}

void FabricMaterial::createTexture(const omni::fabric::Path& path) {
    return createTextureCommon(path, FabricTokens::cesium_internal_texture_lookup);
}

void FabricMaterial::createRasterOverlayLayer(const omni::fabric::Path& path) {
    static const auto additionalAttributes = std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>{{
        std::make_pair(FabricTypes::inputs_alpha, FabricTokens::inputs_alpha),
    }};
    return createTextureCommon(path, FabricTokens::cesium_internal_imagery_layer_lookup, additionalAttributes);
}

void FabricMaterial::createRasterOverlayLayerResolverCommon(
    const omni::fabric::Path& path,
    uint64_t rasterOverlayLayerCount,
    const omni::fabric::Token& subidentifier) {
    auto& fabricStage = _pContext->getFabricStage();

    fabricStage.createPrim(path);

    FabricAttributesBuilder attributes(_pContext);

    attributes.addAttribute(FabricTypes::inputs_imagery_layers_count, FabricTokens::inputs_imagery_layers_count);

    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);

    const auto rasterOverlayLayerCountFabric =
        fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_imagery_layers_count);
    *rasterOverlayLayerCountFabric = static_cast<int>(rasterOverlayLayerCount);
}

void FabricMaterial::createRasterOverlayLayerResolver(const omni::fabric::Path& path, uint64_t rasterOverlayLayerCount) {
    createRasterOverlayLayerResolverCommon(path, rasterOverlayLayerCount, FabricTokens::cesium_internal_imagery_layer_resolver);
}

void FabricMaterial::createClippingRasterOverlayLayerResolver(
    const omni::fabric::Path& path,
    uint64_t clippingRasterOverlayLayerCount) {
    createRasterOverlayLayerResolverCommon(
        path, clippingRasterOverlayLayerCount, FabricTokens::cesium_internal_clipping_imagery_layer_resolver);
}

void FabricMaterial::createFeatureIdIndex(const omni::fabric::Path& path) {
    createFeatureIdAttribute(path);
}

void FabricMaterial::createFeatureIdAttribute(const omni::fabric::Path& path) {
    auto& fabricStage = _pContext->getFabricStage();

    fabricStage.createPrim(path);

    FabricAttributesBuilder attributes(_pContext);

    attributes.addAttribute(FabricTypes::inputs_primvar_name, FabricTokens::inputs_primvar_name);
    attributes.addAttribute(FabricTypes::inputs_null_feature_id, FabricTokens::inputs_null_feature_id);

    createAttributes(
        *_pContext, fabricStage, path, attributes, FabricTokens::cesium_internal_feature_id_attribute_lookup);
}

void FabricMaterial::createFeatureIdTexture(const omni::fabric::Path& path) {
    static const auto additionalAttributes = std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>{{
        std::make_pair(FabricTypes::inputs_channels, FabricTokens::inputs_channels),
        std::make_pair(FabricTypes::inputs_channel_count, FabricTokens::inputs_channel_count),
        std::make_pair(FabricTypes::inputs_null_feature_id, FabricTokens::inputs_null_feature_id),
    }};

    return createTextureCommon(path, FabricTokens::cesium_internal_feature_id_texture_lookup, additionalAttributes);
}

void FabricMaterial::createPropertyAttributePropertyInt(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::inputs_primvar_name, FabricTokens::inputs_primvar_name);
    attributes.addAttribute(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data);
    attributes.addAttribute(noDataType, FabricTokens::inputs_no_data);
    attributes.addAttribute(defaultValueType, FabricTokens::inputs_default_value);
    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);
}

void FabricMaterial::createPropertyAttributePropertyNormalizedInt(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType,
    const omni::fabric::Type& offsetType,
    const omni::fabric::Type& scaleType,
    const omni::fabric::Type& maximumValueType) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::inputs_primvar_name, FabricTokens::inputs_primvar_name);
    attributes.addAttribute(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data);
    attributes.addAttribute(noDataType, FabricTokens::inputs_no_data);
    attributes.addAttribute(defaultValueType, FabricTokens::inputs_default_value);
    attributes.addAttribute(offsetType, FabricTokens::inputs_offset);
    attributes.addAttribute(scaleType, FabricTokens::inputs_scale);
    attributes.addAttribute(maximumValueType, FabricTokens::inputs_maximum_value);
    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);
}

void FabricMaterial::createPropertyAttributePropertyFloat(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType,
    const omni::fabric::Type& offsetType,
    const omni::fabric::Type& scaleType) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::inputs_primvar_name, FabricTokens::inputs_primvar_name);
    attributes.addAttribute(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data);
    attributes.addAttribute(noDataType, FabricTokens::inputs_no_data);
    attributes.addAttribute(defaultValueType, FabricTokens::inputs_default_value);
    attributes.addAttribute(offsetType, FabricTokens::inputs_offset);
    attributes.addAttribute(scaleType, FabricTokens::inputs_scale);
    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);
}

void FabricMaterial::createPropertyAttributeProperty(const omni::fabric::Path& path, MdlInternalPropertyType type) {
    switch (type) {
        case MdlInternalPropertyType::INT32:
            createPropertyAttributePropertyInt(
                path,
                FabricTokens::cesium_internal_property_attribute_int_lookup,
                FabricTypes::inputs_no_data_int,
                FabricTypes::inputs_default_value_int);
            break;
        case MdlInternalPropertyType::VEC2_INT32:
            createPropertyAttributePropertyInt(
                path,
                FabricTokens::cesium_internal_property_attribute_int2_lookup,
                FabricTypes::inputs_no_data_int2,
                FabricTypes::inputs_default_value_int2);
            break;
        case MdlInternalPropertyType::VEC3_INT32:
            createPropertyAttributePropertyInt(
                path,
                FabricTokens::cesium_internal_property_attribute_int3_lookup,
                FabricTypes::inputs_no_data_int3,
                FabricTypes::inputs_default_value_int3);
            break;
        case MdlInternalPropertyType::VEC4_INT32:
            createPropertyAttributePropertyInt(
                path,
                FabricTokens::cesium_internal_property_attribute_int4_lookup,
                FabricTypes::inputs_no_data_int4,
                FabricTypes::inputs_default_value_int4);
            break;
        case MdlInternalPropertyType::INT32_NORM:
            createPropertyAttributePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_attribute_normalized_int_lookup,
                FabricTypes::inputs_no_data_int,
                FabricTypes::inputs_default_value_float,
                FabricTypes::inputs_offset_float,
                FabricTypes::inputs_scale_float,
                FabricTypes::inputs_maximum_value_int);
            break;
        case MdlInternalPropertyType::VEC2_INT32_NORM:
            createPropertyAttributePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_attribute_normalized_int2_lookup,
                FabricTypes::inputs_no_data_int2,
                FabricTypes::inputs_default_value_float2,
                FabricTypes::inputs_offset_float2,
                FabricTypes::inputs_scale_float2,
                FabricTypes::inputs_maximum_value_int2);
            break;
        case MdlInternalPropertyType::VEC3_INT32_NORM:
            createPropertyAttributePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_attribute_normalized_int3_lookup,
                FabricTypes::inputs_no_data_int3,
                FabricTypes::inputs_default_value_float3,
                FabricTypes::inputs_offset_float3,
                FabricTypes::inputs_scale_float3,
                FabricTypes::inputs_maximum_value_int3);
            break;
        case MdlInternalPropertyType::VEC4_INT32_NORM:
            createPropertyAttributePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_attribute_normalized_int4_lookup,
                FabricTypes::inputs_no_data_int4,
                FabricTypes::inputs_default_value_float4,
                FabricTypes::inputs_offset_float4,
                FabricTypes::inputs_scale_float4,
                FabricTypes::inputs_maximum_value_int4);
            break;
        case MdlInternalPropertyType::FLOAT32:
            createPropertyAttributePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_attribute_float_lookup,
                FabricTypes::inputs_no_data_float,
                FabricTypes::inputs_default_value_float,
                FabricTypes::inputs_offset_float,
                FabricTypes::inputs_scale_float);
            break;
        case MdlInternalPropertyType::VEC2_FLOAT32:
            createPropertyAttributePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_attribute_float2_lookup,
                FabricTypes::inputs_no_data_float2,
                FabricTypes::inputs_default_value_float2,
                FabricTypes::inputs_offset_float2,
                FabricTypes::inputs_scale_float2);
            break;
        case MdlInternalPropertyType::VEC3_FLOAT32:
            createPropertyAttributePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_attribute_float3_lookup,
                FabricTypes::inputs_no_data_float3,
                FabricTypes::inputs_default_value_float3,
                FabricTypes::inputs_offset_float3,
                FabricTypes::inputs_scale_float3);
            break;
        case MdlInternalPropertyType::VEC4_FLOAT32:
            createPropertyAttributePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_attribute_float4_lookup,
                FabricTypes::inputs_no_data_float4,
                FabricTypes::inputs_default_value_float4,
                FabricTypes::inputs_offset_float4,
                FabricTypes::inputs_scale_float4);
            break;
        case MdlInternalPropertyType::MAT2_INT32:
        case MdlInternalPropertyType::MAT2_FLOAT32:
        case MdlInternalPropertyType::MAT2_INT32_NORM:
        case MdlInternalPropertyType::MAT3_INT32:
        case MdlInternalPropertyType::MAT3_FLOAT32:
        case MdlInternalPropertyType::MAT3_INT32_NORM:
        case MdlInternalPropertyType::MAT4_INT32:
        case MdlInternalPropertyType::MAT4_FLOAT32:
        case MdlInternalPropertyType::MAT4_INT32_NORM:
            break;
    }
}

void FabricMaterial::createPropertyTexturePropertyInt(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType) {
    static const auto additionalAttributes = std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>{{
        std::make_pair(FabricTypes::inputs_channels, FabricTokens::inputs_channels),
        std::make_pair(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data),
        std::make_pair(noDataType, FabricTokens::inputs_no_data),
        std::make_pair(defaultValueType, FabricTokens::inputs_default_value),
    }};
    return createTextureCommon(path, subidentifier, additionalAttributes);
}

void FabricMaterial::createPropertyTexturePropertyNormalizedInt(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType,
    const omni::fabric::Type& offsetType,
    const omni::fabric::Type& scaleType,
    const omni::fabric::Type& maximumValueType) {
    static const auto additionalAttributes = std::vector<std::pair<omni::fabric::Type, omni::fabric::Token>>{{
        std::make_pair(FabricTypes::inputs_channels, FabricTokens::inputs_channels),
        std::make_pair(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data),
        std::make_pair(noDataType, FabricTokens::inputs_no_data),
        std::make_pair(defaultValueType, FabricTokens::inputs_default_value),
        std::make_pair(offsetType, FabricTokens::inputs_offset),
        std::make_pair(scaleType, FabricTokens::inputs_scale),
        std::make_pair(maximumValueType, FabricTokens::inputs_maximum_value),
    }};

    return createTextureCommon(path, subidentifier, additionalAttributes);
}

void FabricMaterial::createPropertyTextureProperty(const omni::fabric::Path& path, MdlInternalPropertyType type) {
    switch (type) {
        case MdlInternalPropertyType::INT32:
            createPropertyTexturePropertyInt(
                path,
                FabricTokens::cesium_internal_property_texture_int_lookup,
                FabricTypes::inputs_no_data_int,
                FabricTypes::inputs_default_value_int);
            break;
        case MdlInternalPropertyType::VEC2_INT32:
            createPropertyTexturePropertyInt(
                path,
                FabricTokens::cesium_internal_property_texture_int2_lookup,
                FabricTypes::inputs_no_data_int2,
                FabricTypes::inputs_default_value_int2);
            break;
        case MdlInternalPropertyType::VEC3_INT32:
            createPropertyTexturePropertyInt(
                path,
                FabricTokens::cesium_internal_property_texture_int3_lookup,
                FabricTypes::inputs_no_data_int3,
                FabricTypes::inputs_default_value_int3);
            break;
        case MdlInternalPropertyType::VEC4_INT32:
            createPropertyTexturePropertyInt(
                path,
                FabricTokens::cesium_internal_property_texture_int4_lookup,
                FabricTypes::inputs_no_data_int4,
                FabricTypes::inputs_default_value_int4);
            break;
        case MdlInternalPropertyType::INT32_NORM:
            createPropertyTexturePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_texture_normalized_int_lookup,
                FabricTypes::inputs_no_data_int,
                FabricTypes::inputs_default_value_float,
                FabricTypes::inputs_offset_float,
                FabricTypes::inputs_scale_float,
                FabricTypes::inputs_maximum_value_int);
            break;
        case MdlInternalPropertyType::VEC2_INT32_NORM:
            createPropertyTexturePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_texture_normalized_int2_lookup,
                FabricTypes::inputs_no_data_int2,
                FabricTypes::inputs_default_value_float2,
                FabricTypes::inputs_offset_float2,
                FabricTypes::inputs_scale_float2,
                FabricTypes::inputs_maximum_value_int2);
            break;
        case MdlInternalPropertyType::VEC3_INT32_NORM:
            createPropertyTexturePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_texture_normalized_int3_lookup,
                FabricTypes::inputs_no_data_int3,
                FabricTypes::inputs_default_value_float3,
                FabricTypes::inputs_offset_float3,
                FabricTypes::inputs_scale_float3,
                FabricTypes::inputs_maximum_value_int3);
            break;
        case MdlInternalPropertyType::VEC4_INT32_NORM:
            createPropertyTexturePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_texture_normalized_int4_lookup,
                FabricTypes::inputs_no_data_int4,
                FabricTypes::inputs_default_value_float4,
                FabricTypes::inputs_offset_float4,
                FabricTypes::inputs_scale_float4,
                FabricTypes::inputs_maximum_value_int4);
            break;
        case MdlInternalPropertyType::FLOAT32:
        case MdlInternalPropertyType::VEC2_FLOAT32:
        case MdlInternalPropertyType::VEC3_FLOAT32:
        case MdlInternalPropertyType::VEC4_FLOAT32:
        case MdlInternalPropertyType::MAT2_INT32:
        case MdlInternalPropertyType::MAT2_FLOAT32:
        case MdlInternalPropertyType::MAT2_INT32_NORM:
        case MdlInternalPropertyType::MAT3_INT32:
        case MdlInternalPropertyType::MAT3_FLOAT32:
        case MdlInternalPropertyType::MAT3_INT32_NORM:
        case MdlInternalPropertyType::MAT4_INT32:
        case MdlInternalPropertyType::MAT4_FLOAT32:
        case MdlInternalPropertyType::MAT4_INT32_NORM:
            break;
    }
}

void FabricMaterial::createPropertyTablePropertyInt(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::inputs_property_table_texture, FabricTokens::inputs_property_table_texture);
    attributes.addAttribute(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data);
    attributes.addAttribute(noDataType, FabricTokens::inputs_no_data);
    attributes.addAttribute(defaultValueType, FabricTokens::inputs_default_value);
    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);
}

void FabricMaterial::createPropertyTablePropertyNormalizedInt(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType,
    const omni::fabric::Type& offsetType,
    const omni::fabric::Type& scaleType,
    const omni::fabric::Type& maximumValueType) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::inputs_property_table_texture, FabricTokens::inputs_property_table_texture);
    attributes.addAttribute(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data);
    attributes.addAttribute(noDataType, FabricTokens::inputs_no_data);
    attributes.addAttribute(defaultValueType, FabricTokens::inputs_default_value);
    attributes.addAttribute(offsetType, FabricTokens::inputs_offset);
    attributes.addAttribute(scaleType, FabricTokens::inputs_scale);
    attributes.addAttribute(maximumValueType, FabricTokens::inputs_maximum_value);
    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);
}

void FabricMaterial::createPropertyTablePropertyFloat(
    const omni::fabric::Path& path,
    const omni::fabric::Token& subidentifier,
    const omni::fabric::Type& noDataType,
    const omni::fabric::Type& defaultValueType,
    const omni::fabric::Type& offsetType,
    const omni::fabric::Type& scaleType) {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(path);
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::inputs_property_table_texture, FabricTokens::inputs_property_table_texture);
    attributes.addAttribute(FabricTypes::inputs_has_no_data, FabricTokens::inputs_has_no_data);
    attributes.addAttribute(noDataType, FabricTokens::inputs_no_data);
    attributes.addAttribute(defaultValueType, FabricTokens::inputs_default_value);
    attributes.addAttribute(offsetType, FabricTokens::inputs_offset);
    attributes.addAttribute(scaleType, FabricTokens::inputs_scale);
    createAttributes(*_pContext, fabricStage, path, attributes, subidentifier);
}

void FabricMaterial::createPropertyTableProperty(const omni::fabric::Path& path, MdlInternalPropertyType type) {
    switch (type) {
        case MdlInternalPropertyType::INT32:
            createPropertyTablePropertyInt(
                path,
                FabricTokens::cesium_internal_property_table_int_lookup,
                FabricTypes::inputs_no_data_int,
                FabricTypes::inputs_default_value_int);
            break;
        case MdlInternalPropertyType::VEC2_INT32:
            createPropertyTablePropertyInt(
                path,
                FabricTokens::cesium_internal_property_table_int2_lookup,
                FabricTypes::inputs_no_data_int2,
                FabricTypes::inputs_default_value_int2);
            break;
        case MdlInternalPropertyType::VEC3_INT32:
            createPropertyTablePropertyInt(
                path,
                FabricTokens::cesium_internal_property_table_int3_lookup,
                FabricTypes::inputs_no_data_int3,
                FabricTypes::inputs_default_value_int3);
            break;
        case MdlInternalPropertyType::VEC4_INT32:
            createPropertyTablePropertyInt(
                path,
                FabricTokens::cesium_internal_property_table_int4_lookup,
                FabricTypes::inputs_no_data_int4,
                FabricTypes::inputs_default_value_int4);
            break;
        case MdlInternalPropertyType::INT32_NORM:
            createPropertyTablePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_table_normalized_int_lookup,
                FabricTypes::inputs_no_data_int,
                FabricTypes::inputs_default_value_float,
                FabricTypes::inputs_offset_float,
                FabricTypes::inputs_scale_float,
                FabricTypes::inputs_maximum_value_int);
            break;
        case MdlInternalPropertyType::VEC2_INT32_NORM:
            createPropertyTablePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_table_normalized_int2_lookup,
                FabricTypes::inputs_no_data_int2,
                FabricTypes::inputs_default_value_float2,
                FabricTypes::inputs_offset_float2,
                FabricTypes::inputs_scale_float2,
                FabricTypes::inputs_maximum_value_int2);
            break;
        case MdlInternalPropertyType::VEC3_INT32_NORM:
            createPropertyTablePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_table_normalized_int3_lookup,
                FabricTypes::inputs_no_data_int3,
                FabricTypes::inputs_default_value_float3,
                FabricTypes::inputs_offset_float3,
                FabricTypes::inputs_scale_float3,
                FabricTypes::inputs_maximum_value_int3);
            break;
        case MdlInternalPropertyType::VEC4_INT32_NORM:
            createPropertyTablePropertyNormalizedInt(
                path,
                FabricTokens::cesium_internal_property_table_normalized_int4_lookup,
                FabricTypes::inputs_no_data_int4,
                FabricTypes::inputs_default_value_float4,
                FabricTypes::inputs_offset_float4,
                FabricTypes::inputs_scale_float4,
                FabricTypes::inputs_maximum_value_int4);
            break;
        case MdlInternalPropertyType::FLOAT32:
            createPropertyTablePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_table_float_lookup,
                FabricTypes::inputs_no_data_float,
                FabricTypes::inputs_default_value_float,
                FabricTypes::inputs_offset_float,
                FabricTypes::inputs_scale_float);
            break;
        case MdlInternalPropertyType::VEC2_FLOAT32:
            createPropertyTablePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_table_float2_lookup,
                FabricTypes::inputs_no_data_float2,
                FabricTypes::inputs_default_value_float2,
                FabricTypes::inputs_offset_float2,
                FabricTypes::inputs_scale_float2);
            break;
        case MdlInternalPropertyType::VEC3_FLOAT32:
            createPropertyTablePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_table_float3_lookup,
                FabricTypes::inputs_no_data_float3,
                FabricTypes::inputs_default_value_float3,
                FabricTypes::inputs_offset_float3,
                FabricTypes::inputs_scale_float3);
            break;
        case MdlInternalPropertyType::VEC4_FLOAT32:
            createPropertyTablePropertyFloat(
                path,
                FabricTokens::cesium_internal_property_table_float4_lookup,
                FabricTypes::inputs_no_data_float4,
                FabricTypes::inputs_default_value_float4,
                FabricTypes::inputs_offset_float4,
                FabricTypes::inputs_scale_float4);
            break;
        case MdlInternalPropertyType::MAT2_INT32:
        case MdlInternalPropertyType::MAT2_FLOAT32:
        case MdlInternalPropertyType::MAT2_INT32_NORM:
        case MdlInternalPropertyType::MAT3_INT32:
        case MdlInternalPropertyType::MAT3_FLOAT32:
        case MdlInternalPropertyType::MAT3_INT32_NORM:
        case MdlInternalPropertyType::MAT4_INT32:
        case MdlInternalPropertyType::MAT4_FLOAT32:
        case MdlInternalPropertyType::MAT4_INT32_NORM:
            break;
    }
}

void FabricMaterial::reset() {
    if (_usesDefaultMaterial) {
        setShaderValues(
            _shaderPath, GltfUtil::getDefaultMaterialInfo(), DEFAULT_DISPLAY_COLOR, DEFAULT_DISPLAY_OPACITY);
    }

    if (_materialDescriptor.hasBaseColorTexture()) {
        setTextureValues(
            _baseColorTexturePath,
            _defaultWhiteTextureAssetPathToken,
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

    for (const auto& [type, paths] : _propertyAttributePropertyPaths) {
        for (const auto& path : paths) {
            CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE(
                clearPropertyAttributeProperty, type, _pContext->getFabricStage(), path);
        }
    }

    for (const auto& [type, paths] : _propertyTexturePropertyPaths) {
        for (const auto& path : paths) {
            CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE(
                clearPropertyTextureProperty,
                type,
                _pContext->getFabricStage(),
                path,
                _defaultTransparentTextureAssetPathToken);
        }
    }

    for (const auto& [type, paths] : _propertyTablePropertyPaths) {
        for (const auto& path : paths) {
            CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE(
                clearPropertyTableProperty,
                type,
                _pContext->getFabricStage(),
                path,
                _defaultTransparentTextureAssetPathToken);
        }
    }

    for (const auto& rasterOverlayLayerPath : _rasterOverlayLayerPaths) {
        setRasterOverlayLayerValues(
            rasterOverlayLayerPath,
            _defaultTransparentTextureAssetPathToken,
            GltfUtil::getDefaultTextureInfo(),
            DEFAULT_TEXCOORD_INDEX,
            DEFAULT_ALPHA);
    }

    for (const auto& path : _allPaths) {
        auto& fabricStage = _pContext->getFabricStage();
        const auto tilesetIdFabric = fabricStage.getAttributeWr<int64_t>(path, FabricTokens::_cesium_tilesetId);
        *tilesetIdFabric = FabricUtil::NO_TILESET_ID;
    }
}

void FabricMaterial::setMaterial(
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
    const std::unordered_map<uint64_t, uint64_t>& propertyTextureIndexMapping) {

    if (stageDestroyed()) {
        return;
    }

    if (_usesDefaultMaterial) {
        _alphaMode = getInitialAlphaMode(_materialDescriptor, materialInfo);

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

    if (_materialDescriptor.hasBaseColorTexture()) {
        const auto& textureInfo = materialInfo.baseColorTexture.value();
        const auto& textureAssetPath = pBaseColorTexture->getAssetPathToken();
        const auto texcoordIndex = texcoordIndexMapping.at(textureInfo.setIndex);

        setTextureValues(_baseColorTexturePath, textureAssetPath, textureInfo, texcoordIndex);
    }

    const auto featureIdCounts = getFeatureIdCounts(_materialDescriptor);

    for (uint64_t i = 0; i < featureIdCounts.indexCount; ++i) {
        const auto featureIdSetIndex = featureIdIndexSetIndexMapping[i];
        const auto featureId = featuresInfo.featureIds[featureIdSetIndex];
        const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
        const auto nullFeatureId = CppUtil::defaultValue(featureId.nullFeatureId, DEFAULT_NULL_FEATURE_ID);

        setFeatureIdIndexValues(featureIdPath, nullFeatureId);
    }

    for (uint64_t i = 0; i < featureIdCounts.attributeCount; ++i) {
        const auto featureIdSetIndex = featureIdAttributeSetIndexMapping[i];
        const auto featureId = featuresInfo.featureIds[featureIdSetIndex];
        const auto attributeSetIndex = std::get<uint64_t>(featureId.featureIdStorage);
        const auto attributeName = fmt::format("_FEATURE_ID_{}", attributeSetIndex);
        const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
        const auto nullFeatureId = CppUtil::defaultValue(featureId.nullFeatureId, DEFAULT_NULL_FEATURE_ID);

        setFeatureIdAttributeValues(featureIdPath, attributeName, nullFeatureId);
    }

    for (uint64_t i = 0; i < featureIdCounts.textureCount; ++i) {
        const auto featureIdSetIndex = featureIdTextureSetIndexMapping[i];
        const auto& featureId = featuresInfo.featureIds[featureIdSetIndex];
        const auto& textureInfo = std::get<FabricTextureInfo>(featureId.featureIdStorage);
        const auto& textureAssetPath = featureIdTextures[i]->getAssetPathToken();
        const auto texcoordIndex = texcoordIndexMapping.at(textureInfo.setIndex);
        const auto& featureIdPath = _featureIdPaths[featureIdSetIndex];
        const auto nullFeatureId = CppUtil::defaultValue(featureId.nullFeatureId, DEFAULT_NULL_FEATURE_ID);

        setFeatureIdTextureValues(featureIdPath, textureAssetPath, textureInfo, texcoordIndex, nullFeatureId);
    }

    const auto& properties = _materialDescriptor.getStyleableProperties();

    const auto getPropertyPath = [this, &properties](const std::string& propertyId) {
        const auto index = CppUtil::indexOfByMember(properties, &FabricPropertyDescriptor::propertyId, propertyId);
        assert(index != properties.size());
        return _propertyPaths[index];
    };

    MetadataUtil::forEachStyleablePropertyAttributeProperty(
        *_pContext,
        model,
        primitive,
        false,
        [this, &getPropertyPath](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyAttributePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<type>();
            const auto& primvarName = property.attribute;
            const auto& propertyPath = getPropertyPath(propertyId);
            const auto& propertyInfo = property.propertyInfo;
            const auto hasNoData = propertyInfo.noData.has_value();
            const auto offset = getOffset(propertyInfo);
            const auto scale = getScale(propertyInfo);
            const auto noData = getNoData(propertyInfo);
            const auto defaultValue = getDefaultValue(propertyInfo);
            constexpr auto maximumValue = getMaximumValue<type>();

            setPropertyAttributePropertyValues<mdlType>(
                _pContext->getFabricStage(),
                propertyPath,
                primvarName,
                offset,
                scale,
                maximumValue,
                hasNoData,
                noData,
                defaultValue);
        });

    MetadataUtil::forEachStyleablePropertyTextureProperty(
        *_pContext,
        model,
        primitive,
        false,
        [this, &propertyTextures, &texcoordIndexMapping, &propertyTextureIndexMapping, &getPropertyPath](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTexturePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<type>();
            const auto& textureInfo = property.textureInfo;
            const auto textureIndex = property.textureIndex;
            const auto& propertyPath = getPropertyPath(propertyId);
            const auto texcoordIndex = texcoordIndexMapping.at(textureInfo.setIndex);
            const auto propertyTextureIndex = propertyTextureIndexMapping.at(textureIndex);
            const auto& textureAssetPath = propertyTextures[propertyTextureIndex]->getAssetPathToken();
            const auto& propertyInfo = property.propertyInfo;
            const auto hasNoData = propertyInfo.noData.has_value();
            const auto offset = getOffset(propertyInfo);
            const auto scale = getScale(propertyInfo);
            const auto noData = getNoData(propertyInfo);
            const auto defaultValue = getDefaultValue(propertyInfo);
            constexpr auto maximumValue = getMaximumValue<type>();

            setPropertyTexturePropertyValues<mdlType>(
                _pContext->getFabricStage(),
                propertyPath,
                textureAssetPath,
                textureInfo,
                texcoordIndex,
                offset,
                scale,
                maximumValue,
                hasNoData,
                noData,
                defaultValue);
        });

    uint64_t propertyTablePropertyCounter = 0;

    MetadataUtil::forEachStyleablePropertyTableProperty(
        *_pContext,
        model,
        primitive,
        false,
        [this, &propertyTableTextures, &propertyTablePropertyCounter, &getPropertyPath](
            const std::string& propertyId,
            [[maybe_unused]] const auto& propertyTablePropertyView,
            const auto& property) {
            constexpr auto type = std::decay_t<decltype(property)>::Type;
            constexpr auto mdlType = DataTypeUtil::getMdlInternalPropertyType<type>();
            const auto& propertyPath = getPropertyPath(propertyId);
            const auto textureIndex = propertyTablePropertyCounter++;
            const auto& textureAssetPath = propertyTableTextures[textureIndex]->getAssetPathToken();
            const auto& propertyInfo = property.propertyInfo;
            const auto hasNoData = propertyInfo.noData.has_value();
            const auto offset = getOffset(propertyInfo);
            const auto scale = getScale(propertyInfo);
            const auto noData = getNoData(propertyInfo);
            const auto defaultValue = getDefaultValue(propertyInfo);
            constexpr auto maximumValue = getMaximumValue<type>();

            setPropertyTablePropertyValues<mdlType>(
                _pContext->getFabricStage(),
                propertyPath,
                textureAssetPath,
                offset,
                scale,
                maximumValue,
                hasNoData,
                noData,
                defaultValue);
        });

    for (const auto& path : _allPaths) {
        auto& fabricStage = _pContext->getFabricStage();
        const auto tilesetIdFabric = fabricStage.getAttributeWr<int64_t>(path, FabricTokens::_cesium_tilesetId);
        *tilesetIdFabric = tilesetId;
    }
}

void FabricMaterial::createConnectionsToCopiedPaths() {
    auto& fabricStage = _pContext->getFabricStage();

    const auto hasBaseColorTexture = _materialDescriptor.hasBaseColorTexture();
    const auto rasterOverlay = getRasterOverlayLayerCount(_materialDescriptor);
    const auto featureIdCount = getFeatureIdCounts(_materialDescriptor).totalCount;

    for (const auto& copiedPath : _copiedBaseColorTexturePaths) {
        if (hasBaseColorTexture) {
            createConnection(fabricStage, _baseColorTexturePath, copiedPath, FabricTokens::inputs_base_color_texture);
        }
    }

    for (const auto& copiedPath : _copiedRasterOverlayLayerPaths) {
        const auto indexFabric = fabricStage.getAttributeRd<int>(copiedPath, FabricTokens::inputs_imagery_layer_index);
        const auto index = static_cast<uint64_t>(CppUtil::defaultValue(indexFabric, 0));

        if (index < rasterOverlay) {
            createConnection(fabricStage, _rasterOverlayLayerPaths[index], copiedPath, FabricTokens::inputs_imagery_layer);
        }
    }

    for (const auto& copiedPath : _copiedFeatureIdPaths) {
        const auto indexFabric = fabricStage.getAttributeRd<int>(copiedPath, FabricTokens::inputs_feature_id_set_index);
        const auto index = static_cast<uint64_t>(CppUtil::defaultValue(indexFabric, 0));

        if (index < featureIdCount) {
            createConnection(fabricStage, _featureIdPaths[index], copiedPath, FabricTokens::inputs_feature_id);
        }
    }
}

void FabricMaterial::destroyConnectionsToCopiedPaths() {
    auto& fabricStage = _pContext->getFabricStage();

    for (const auto& copiedPath : _copiedBaseColorTexturePaths) {
        destroyConnection(fabricStage, copiedPath, FabricTokens::inputs_base_color_texture);
    }

    for (const auto& copiedPath : _copiedRasterOverlayLayerPaths) {
        destroyConnection(fabricStage, copiedPath, FabricTokens::inputs_imagery_layer);
    }

    for (const auto& copiedPath : _copiedFeatureIdPaths) {
        destroyConnection(fabricStage, copiedPath, FabricTokens::inputs_feature_id);
    }
}

void FabricMaterial::createConnectionsToProperties() {
    auto& fabricStage = _pContext->getFabricStage();
    const auto& properties = _materialDescriptor.getStyleableProperties();

    for (const auto& propertyPathExternal : _copiedPropertyPaths) {
        const auto propertyId = getStringFabric(fabricStage, propertyPathExternal, FabricTokens::inputs_property_id);
        const auto mdlIdentifier = FabricUtil::getMdlIdentifier(fabricStage, propertyPathExternal);
        const auto propertyTypeExternal = FabricUtil::getMdlExternalPropertyType(mdlIdentifier);

        const auto index = CppUtil::indexOfByMember(properties, &FabricPropertyDescriptor::propertyId, propertyId);

        if (index == properties.size()) {
            _pContext->getLogger()->warn(
                "Could not find property \"{}\" referenced by {}. A default value will be returned instead.",
                propertyId,
                mdlIdentifier.getText());
            continue;
        }

        const auto propertyTypeInternal = properties[index].type;

        if (!FabricUtil::typesCompatible(propertyTypeExternal, propertyTypeInternal)) {
            _pContext->getLogger()->warn(
                "Property \"{}\" referenced by {} has incompatible type. A default value will be returned instead.",
                propertyId,
                mdlIdentifier.getText());
            continue;
        }

        const auto& propertyPathInternal = _propertyPaths[index];

        createConnection(fabricStage, propertyPathInternal, propertyPathExternal, FabricTokens::inputs_property_value);
    }
}

void FabricMaterial::destroyConnectionsToProperties() {
    auto& fabricStage = _pContext->getFabricStage();

    for (const auto& copiedPath : _copiedPropertyPaths) {
        destroyConnection(fabricStage, copiedPath, FabricTokens::inputs_property_value);
    }
}

void FabricMaterial::setRasterOverlayLayer(
    FabricTexture* pTexture,
    const FabricTextureInfo& textureInfo,
    uint64_t rasterOverlayLayerIndex,
    double alpha,
    const std::unordered_map<uint64_t, uint64_t>& rasterOverlayTexcoordIndexMapping) {
    if (stageDestroyed()) {
        return;
    }

    if (rasterOverlayLayerIndex >= _rasterOverlayLayerPaths.size()) {
        return;
    }

    const auto& textureAssetPath = pTexture->getAssetPathToken();
    const auto texcoordIndex = rasterOverlayTexcoordIndexMapping.at(textureInfo.setIndex);
    const auto& rasterOverlay = _rasterOverlayLayerPaths[rasterOverlayLayerIndex];
    setRasterOverlayLayerValues(rasterOverlay, textureAssetPath, textureInfo, texcoordIndex, alpha);
}

void FabricMaterial::setRasterOverlayLayerAlpha(uint64_t rasterOverlayLayerIndex, double alpha) {
    if (stageDestroyed()) {
        return;
    }

    if (rasterOverlayLayerIndex >= _rasterOverlayLayerPaths.size()) {
        return;
    }

    const auto& rasterOverlayLayerPath = _rasterOverlayLayerPaths[rasterOverlayLayerIndex];
    setRasterOverlayLayerAlphaValue(rasterOverlayLayerPath, alpha);
}

void FabricMaterial::setDisplayColorAndOpacity(const glm::dvec3& displayColor, double displayOpacity) {
    if (stageDestroyed()) {
        return;
    }

    if (!_usesDefaultMaterial) {
        return;
    }

    auto& fabricStage = _pContext->getFabricStage();

    const auto tileColorFabric = fabricStage.getAttributeWr<glm::fvec4>(_shaderPath, FabricTokens::inputs_tile_color);
    const auto alphaModeFabric = fabricStage.getAttributeWr<int>(_shaderPath, FabricTokens::inputs_alpha_mode);

    *tileColorFabric = glm::fvec4(getTileColor(_debugColor, displayColor, displayOpacity));
    *alphaModeFabric = getAlphaMode(_alphaMode, displayOpacity);
}

void FabricMaterial::updateShaderInput(const omni::fabric::Path& path, const omni::fabric::Token& attributeName) {
    if (stageDestroyed()) {
        return;
    }

    auto& fabricStage = _pContext->getFabricStage();
    const auto iFabricStage = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();

    const auto copiedShaderPath = FabricUtil::getCopiedShaderPath(_materialPath, path);
    const auto attributesToCopy = std::vector<omni::fabric::TokenC>{attributeName.asTokenC()};

    assert(fabricStage.primExists(copiedShaderPath));

    iFabricStage->copySpecifiedAttributes(
        fabricStage.getId(),
        path,
        attributesToCopy.data(),
        copiedShaderPath,
        attributesToCopy.data(),
        attributesToCopy.size());

    if (attributeName == FabricTokens::inputs_imagery_layer_index ||
        attributeName == FabricTokens::inputs_feature_id_set_index) {
        destroyConnectionsToCopiedPaths();
        createConnectionsToCopiedPaths();
    }

    if (attributeName == FabricTokens::inputs_property_id) {
        destroyConnectionsToProperties();
        createConnectionsToProperties();
    }
}

void FabricMaterial::clearRasterOverlayLayer(uint64_t rasterOverlayLayerIndex) {
    if (stageDestroyed()) {
        return;
    }

    if (rasterOverlayLayerIndex >= _rasterOverlayLayerPaths.size()) {
        return;
    }

    const auto& rasterOverlayLayerPath = _rasterOverlayLayerPaths[rasterOverlayLayerIndex];
    setRasterOverlayLayerValues(
        rasterOverlayLayerPath,
        _defaultTransparentTextureAssetPathToken,
        GltfUtil::getDefaultTextureInfo(),
        DEFAULT_TEXCOORD_INDEX,
        DEFAULT_ALPHA);
}

void FabricMaterial::setShaderValues(
    const omni::fabric::Path& path,
    const FabricMaterialInfo& materialInfo,
    const glm::dvec3& displayColor,
    double displayOpacity) {
    auto& fabricStage = _pContext->getFabricStage();

    const auto tileColorFabric = fabricStage.getAttributeWr<pxr::GfVec4f>(path, FabricTokens::inputs_tile_color);
    const auto alphaCutoffFabric = fabricStage.getAttributeWr<float>(path, FabricTokens::inputs_alpha_cutoff);
    const auto alphaModeFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_alpha_mode);
    const auto baseAlphaFabric = fabricStage.getAttributeWr<float>(path, FabricTokens::inputs_base_alpha);
    const auto baseColorFactorFabric =
        fabricStage.getAttributeWr<pxr::GfVec3f>(path, FabricTokens::inputs_base_color_factor);
    const auto emissiveFactorFabric =
        fabricStage.getAttributeWr<pxr::GfVec3f>(path, FabricTokens::inputs_emissive_factor);
    const auto metallicFactorFabric = fabricStage.getAttributeWr<float>(path, FabricTokens::inputs_metallic_factor);
    const auto roughnessFactorFabric = fabricStage.getAttributeWr<float>(path, FabricTokens::inputs_roughness_factor);

    *tileColorFabric = UsdUtil::glmToUsdVector(glm::fvec4(getTileColor(_debugColor, displayColor, displayOpacity)));
    *alphaCutoffFabric = static_cast<float>(materialInfo.alphaCutoff);
    *alphaModeFabric = getAlphaMode(_alphaMode, displayOpacity);
    *baseAlphaFabric = static_cast<float>(materialInfo.baseAlpha);
    *baseColorFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.baseColorFactor));
    *emissiveFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.emissiveFactor));
    *metallicFactorFabric = static_cast<float>(materialInfo.metallicFactor);
    *roughnessFactorFabric = static_cast<float>(materialInfo.roughnessFactor);
}

void FabricMaterial::setTextureValues(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const FabricTextureInfo& textureInfo,
    uint64_t texcoordIndex) {
    setTextureValuesCommon(_pContext->getFabricStage(), path, textureAssetPathToken, textureInfo, texcoordIndex);
}

void FabricMaterial::setRasterOverlayLayerValues(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const FabricTextureInfo& textureInfo,
    uint64_t texcoordIndex,
    double alpha) {
    setTextureValuesCommon(_pContext->getFabricStage(), path, textureAssetPathToken, textureInfo, texcoordIndex);
    setRasterOverlayLayerAlphaValue(path, alpha);
}

void FabricMaterial::setRasterOverlayLayerAlphaValue(const omni::fabric::Path& path, double alpha) {
    const auto alphaFabric = _pContext->getFabricStage().getAttributeWr<float>(path, FabricTokens::inputs_alpha);
    *alphaFabric = static_cast<float>(alpha);
}

void FabricMaterial::setFeatureIdIndexValues(const omni::fabric::Path& path, int nullFeatureId) {
    setFeatureIdAttributeValues(path, pxr::UsdTokens->vertexId.GetString(), nullFeatureId);
}

void FabricMaterial::setFeatureIdAttributeValues(
    const omni::fabric::Path& path,
    const std::string& primvarName,
    int nullFeatureId) {

    auto& fabricStage = _pContext->getFabricStage();

    setStringFabric(fabricStage, path, FabricTokens::inputs_primvar_name, primvarName);

    const auto nullFeatureIdFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_null_feature_id);
    *nullFeatureIdFabric = nullFeatureId;
}

void FabricMaterial::setFeatureIdTextureValues(
    const omni::fabric::Path& path,
    const pxr::TfToken& textureAssetPathToken,
    const FabricTextureInfo& textureInfo,
    uint64_t texcoordIndex,
    int nullFeatureId) {

    auto& fabricStage = _pContext->getFabricStage();

    setTextureValuesCommonChannels(fabricStage, path, textureAssetPathToken, textureInfo, texcoordIndex);

    const auto nullFeatureIdFabric = fabricStage.getAttributeWr<int>(path, FabricTokens::inputs_null_feature_id);
    *nullFeatureIdFabric = nullFeatureId;
}

bool FabricMaterial::stageDestroyed() {
    // Tile render resources may be processed asynchronously even after the tileset and stage have been destroyed.
    // Add this check to all public member functions, including constructors and destructors, to prevent them from
    // modifying the stage.
    return _stageId != _pContext->getUsdStageId();
}

} // namespace cesium::omniverse
