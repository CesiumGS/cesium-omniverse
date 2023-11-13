#include "cesium/omniverse/GltfUtil.h"

#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/VertexAttributeType.h"

#include <CesiumGltf/FeatureIdTexture.h>

#include <optional>

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/ExtensionExtMeshFeatures.h>
#include <CesiumGltf/ExtensionKhrMaterialsUnlit.h>
#include <CesiumGltf/ExtensionKhrTextureTransform.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltf/TextureInfo.h>
#include <spdlog/fmt/fmt.h>

#include <charconv>
#include <numeric>

namespace cesium::omniverse::GltfUtil {

namespace {

const CesiumGltf::Material defaultMaterial;
const CesiumGltf::MaterialPBRMetallicRoughness defaultPbrMetallicRoughness;
const CesiumGltf::Sampler defaultSampler;
const CesiumGltf::ExtensionKhrTextureTransform defaultTextureTransform;
const CesiumGltf::TextureInfo defaultTextureInfo;

template <typename T, typename U> std::optional<T> castOptional(const std::optional<U>& optional) {
    return optional.has_value() ? std::make_optional(static_cast<T>(optional.value())) : std::nullopt;
}

template <typename IndexType>
IndicesAccessor getIndicesAccessor(
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::AccessorView<IndexType>& indicesAccessorView) {
    if (indicesAccessorView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLES) {
        if (indicesAccessorView.size() % 3 != 0) {
            return {};
        }

        return IndicesAccessor(indicesAccessorView);
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLE_STRIP) {
        if (indicesAccessorView.size() <= 2) {
            return {};
        }

        return IndicesAccessor::FromTriangleStrips(indicesAccessorView);
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLE_FAN) {
        if (indicesAccessorView.size() <= 2) {
            return {};
        }

        return IndicesAccessor::FromTriangleFans(indicesAccessorView);
    }

    return {};
}

CesiumGltf::AccessorView<glm::fvec2> getTexcoordsView(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& semantic,
    uint64_t setIndex) {

    const auto texcoordAttribute = primitive.attributes.find(fmt::format("{}_{}", semantic, setIndex));
    if (texcoordAttribute == primitive.attributes.end()) {
        return {};
    }

    auto texcoordAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, texcoordAttribute->second);
    if (!texcoordAccessor) {
        return {};
    }

    auto texcoordsView = CesiumGltf::AccessorView<glm::fvec2>(model, *texcoordAccessor);

    if (texcoordsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return texcoordsView;
}

CesiumGltf::AccessorView<glm::fvec3>
getNormalsView(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    auto normalAttribute = primitive.attributes.find("NORMAL");
    if (normalAttribute == primitive.attributes.end()) {
        return {};
    }

    auto normalAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, normalAttribute->second);
    if (!normalAccessor) {
        return {};
    }

    const auto normalsView = CesiumGltf::AccessorView<glm::fvec3>(model, *normalAccessor);

    if (normalsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return normalsView;
}

CesiumGltf::AccessorView<glm::fvec3>
getPositionsView(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    auto positionAttribute = primitive.attributes.find("POSITION");
    if (positionAttribute == primitive.attributes.end()) {
        return {};
    }

    auto positionAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, positionAttribute->second);
    if (!positionAccessor) {
        return {};
    }

    auto positionsView = CesiumGltf::AccessorView<glm::fvec3>(model, *positionAccessor);
    if (positionsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return positionsView;
}

TexcoordsAccessor getTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& semantic,
    uint64_t setIndex,
    bool flipVertical) {

    const auto texcoordsView = getTexcoordsView(model, primitive, semantic, setIndex);

    if (texcoordsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return {texcoordsView, flipVertical};
}

template <typename VertexColorType>
VertexColorsAccessor getVertexColorsAccessor(const CesiumGltf::Model& model, const CesiumGltf::Accessor& accessor) {
    CesiumGltf::AccessorView<VertexColorType> view{model, accessor};
    if (view.status() == CesiumGltf::AccessorViewStatus::Valid) {
        return VertexColorsAccessor(view);
    }
    return {};
}

double getAlphaCutoff(const CesiumGltf::Material& material) {
    return material.alphaCutoff;
}

AlphaMode getAlphaMode(const CesiumGltf::Material& material) {
    if (material.alphaMode == CesiumGltf::Material::AlphaMode::OPAQUE) {
        return AlphaMode::OPAQUE;
    } else if (material.alphaMode == CesiumGltf::Material::AlphaMode::MASK) {
        return AlphaMode::MASK;
    } else if (material.alphaMode == CesiumGltf::Material::AlphaMode::BLEND) {
        return AlphaMode::BLEND;
    }

    return AlphaMode::OPAQUE;
}

double getBaseAlpha(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return pbrMetallicRoughness.baseColorFactor[3];
}

glm::dvec3 getBaseColorFactor(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return {
        pbrMetallicRoughness.baseColorFactor[0],
        pbrMetallicRoughness.baseColorFactor[1],
        pbrMetallicRoughness.baseColorFactor[2],
    };
}

glm::dvec3 getEmissiveFactor(const CesiumGltf::Material& material) {
    return {
        material.emissiveFactor[0],
        material.emissiveFactor[1],
        material.emissiveFactor[2],
    };
}

double getMetallicFactor(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return pbrMetallicRoughness.metallicFactor;
}

double getRoughnessFactor(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return pbrMetallicRoughness.roughnessFactor;
}

bool getDoubleSided(const CesiumGltf::Material& material) {
    return material.doubleSided;
}

int32_t getWrapS(const CesiumGltf::Sampler& sampler) {
    return sampler.wrapS;
}

int32_t getWrapT(const CesiumGltf::Sampler& sampler) {
    return sampler.wrapT;
}

glm::dvec2 getTexcoordOffset(const CesiumGltf::ExtensionKhrTextureTransform& textureTransform) {
    const auto& offset = textureTransform.offset;
    return {offset[0], offset[1]};
}

double getTexcoordRotation(const CesiumGltf::ExtensionKhrTextureTransform& textureTransform) {
    return textureTransform.rotation;
}

glm::dvec2 getTexcoordScale(const CesiumGltf::ExtensionKhrTextureTransform& textureTransform) {
    const auto& scale = textureTransform.scale;
    return {scale[0], scale[1]};
}

uint64_t getTexcoordSetIndex(const CesiumGltf::TextureInfo& textureInfo) {
    return static_cast<uint64_t>(textureInfo.index);
}

double getDefaultAlphaCutoff() {
    return getAlphaCutoff(defaultMaterial);
}

AlphaMode getDefaultAlphaMode() {
    return getAlphaMode(defaultMaterial);
}

double getDefaultBaseAlpha() {
    return getBaseAlpha(defaultPbrMetallicRoughness);
}

glm::dvec3 getDefaultBaseColorFactor() {
    return getBaseColorFactor(defaultPbrMetallicRoughness);
}

glm::dvec3 getDefaultEmissiveFactor() {
    return getEmissiveFactor(defaultMaterial);
}

double getDefaultMetallicFactor() {
    return getMetallicFactor(defaultPbrMetallicRoughness);
}

double getDefaultRoughnessFactor() {
    return getRoughnessFactor(defaultPbrMetallicRoughness);
}

bool getDefaultDoubleSided() {
    return getDoubleSided(defaultMaterial);
}

glm::dvec2 getDefaultTexcoordOffset() {
    return getTexcoordOffset(defaultTextureTransform);
}

double getDefaultTexcoordRotation() {
    return getTexcoordRotation(defaultTextureTransform);
}

glm::dvec2 getDefaultTexcoordScale() {
    return getTexcoordScale(defaultTextureTransform);
}

uint64_t getDefaultTexcoordSetIndex() {
    return getTexcoordSetIndex(defaultTextureInfo);
}

int32_t getDefaultWrapS() {
    return getWrapS(defaultSampler);
}

int32_t getDefaultWrapT() {
    return getWrapT(defaultSampler);
}

TextureInfo getTextureInfo(const CesiumGltf::Model& model, const CesiumGltf::TextureInfo& textureInfoGltf) {
    TextureInfo textureInfo = getDefaultTextureInfo();

    textureInfo.setIndex = static_cast<uint64_t>(textureInfoGltf.texCoord);

    if (textureInfoGltf.hasExtension<CesiumGltf::ExtensionKhrTextureTransform>()) {
        const auto& textureTransform = *textureInfoGltf.getExtension<CesiumGltf::ExtensionKhrTextureTransform>();
        textureInfo.offset = getTexcoordOffset(textureTransform);
        textureInfo.rotation = getTexcoordRotation(textureTransform);
        textureInfo.scale = getTexcoordScale(textureTransform);
    }

    const auto index = textureInfoGltf.index;
    if (index >= 0 && static_cast<size_t>(index) < model.textures.size()) {
        const auto& texture = model.textures[static_cast<size_t>(index)];
        const auto samplerIndex = texture.sampler;
        if (samplerIndex != -1) {
            const auto& sampler = model.samplers[static_cast<uint64_t>(samplerIndex)];
            textureInfo.wrapS = getWrapS(sampler);
            textureInfo.wrapT = getWrapT(sampler);
        }
    }

    return textureInfo;
}

TextureInfo
getFeatureIdTextureInfo(const CesiumGltf::Model& model, const CesiumGltf::FeatureIdTexture& featureIdTextureInfo) {
    TextureInfo textureInfo = getTextureInfo(model, featureIdTextureInfo);

    std::vector<uint8_t> channels;
    channels.reserve(featureIdTextureInfo.channels.size());

    for (const auto channel : featureIdTextureInfo.channels) {
        channels.push_back(static_cast<uint8_t>(channel));
    }

    textureInfo.channels = channels;

    return textureInfo;
}

const CesiumGltf::ImageCesium& getImageCesium(const CesiumGltf::Model& model, const CesiumGltf::Texture& texture) {
    const auto imageId = static_cast<uint64_t>(texture.source);
    const auto& image = model.images[imageId];
    return image.cesium;
}

std::pair<std::string, uint64_t> parseAttributeName(const std::string& attributeName) {
    int searchPosition = static_cast<int>(attributeName.size()) - 1;
    int lastUnderscorePosition{-1};
    while (searchPosition > 0) {
        if (!isdigit(attributeName[static_cast<size_t>(searchPosition)])) {
            if (attributeName[static_cast<size_t>(searchPosition)] == '_') {
                lastUnderscorePosition = searchPosition;
            }

            break;
        }
        searchPosition--;
    }

    std::string semantic{};
    uint64_t setIndexU64{0};
    if (lastUnderscorePosition == -1) {
        semantic = attributeName;
    } else {
        semantic = attributeName.substr(0, static_cast<size_t>(lastUnderscorePosition));
        std::from_chars(
            attributeName.data() + lastUnderscorePosition + 1,
            attributeName.data() + attributeName.size(),
            setIndexU64);
    }

    return std::make_pair(semantic, setIndexU64);
}

} // namespace

PositionsAccessor getPositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    const auto positionsView = getPositionsView(model, primitive);

    if (positionsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return {positionsView};
}

std::optional<std::array<glm::dvec3, 2>>
getExtent(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    auto positionAttribute = primitive.attributes.find("POSITION");
    if (positionAttribute == primitive.attributes.end()) {
        return std::nullopt;
    }

    auto positionAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, positionAttribute->second);
    if (!positionAccessor) {
        return std::nullopt;
    }

    const auto& min = positionAccessor->min;
    const auto& max = positionAccessor->max;

    if (min.size() != 3 || max.size() != 3) {
        return std::nullopt;
    }

    return std::array<glm::dvec3, 2>{{glm::dvec3(min[0], min[1], min[2]), glm::dvec3(max[0], max[1], max[2])}};
}

IndicesAccessor getIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PositionsAccessor& positions) {
    const auto indicesAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, primitive.indices);
    if (!indicesAccessor) {
        return {positions.size()};
    }

    if (indicesAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_BYTE) {
        CesiumGltf::AccessorView<uint8_t> view{model, *indicesAccessor};
        return getIndicesAccessor(primitive, view);
    } else if (indicesAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_SHORT) {
        CesiumGltf::AccessorView<uint16_t> view{model, *indicesAccessor};
        return getIndicesAccessor(primitive, view);
    } else if (indicesAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_INT) {
        CesiumGltf::AccessorView<uint32_t> view{model, *indicesAccessor};
        return getIndicesAccessor(primitive, view);
    }

    return {};
}

FaceVertexCountsAccessor getFaceVertexCounts(const IndicesAccessor& indices) {
    return {indices.size() / 3};
}

NormalsAccessor getNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PositionsAccessor& positions,
    const IndicesAccessor& indices,
    bool smoothNormals) {

    const auto normalsView = getNormalsView(model, primitive);

    if (normalsView.status() == CesiumGltf::AccessorViewStatus::Valid) {
        return {normalsView};
    }

    if (smoothNormals) {
        return NormalsAccessor::GenerateSmooth(positions, indices);
    }

    // Otherwise if normals are missing and smoothNormals is false Omniverse will generate flat normals for us automatically
    return {};
}

TexcoordsAccessor
getTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getTexcoords(model, primitive, "TEXCOORD", setIndex, true);
}

TexcoordsAccessor
getImageryTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getTexcoords(model, primitive, "_CESIUMOVERLAY", setIndex, false);
}

VertexColorsAccessor
getVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    const auto vertexColorAttribute = primitive.attributes.find(fmt::format("{}_{}", "COLOR", setIndex));
    if (vertexColorAttribute == primitive.attributes.end()) {
        return {};
    }

    auto vertexColorAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, vertexColorAttribute->second);
    if (!vertexColorAccessor) {
        return {};
    }

    if (vertexColorAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_BYTE) {
        if (vertexColorAccessor->type == CesiumGltf::AccessorSpec::Type::VEC3) {
            return getVertexColorsAccessor<glm::u8vec3>(model, *vertexColorAccessor);
        } else if (vertexColorAccessor->type == CesiumGltf::AccessorSpec::Type::VEC4) {
            return getVertexColorsAccessor<glm::u8vec4>(model, *vertexColorAccessor);
        }
    } else if (vertexColorAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_SHORT) {
        if (vertexColorAccessor->type == CesiumGltf::AccessorSpec::Type::VEC3) {
            return getVertexColorsAccessor<glm::u16vec3>(model, *vertexColorAccessor);
        } else if (vertexColorAccessor->type == CesiumGltf::AccessorSpec::Type::VEC4) {
            return getVertexColorsAccessor<glm::u16vec4>(model, *vertexColorAccessor);
        }
    } else if (vertexColorAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::FLOAT) {
        if (vertexColorAccessor->type == CesiumGltf::AccessorSpec::Type::VEC3) {
            return getVertexColorsAccessor<glm::fvec3>(model, *vertexColorAccessor);
        } else if (vertexColorAccessor->type == CesiumGltf::AccessorSpec::Type::VEC4) {
            return getVertexColorsAccessor<glm::fvec4>(model, *vertexColorAccessor);
        }
    }

    return {};
}

VertexIdsAccessor getVertexIds(const PositionsAccessor& positionsAccessor) {
    return {positionsAccessor.size()};
}

template <VertexAttributeType T>
VertexAttributeAccessor<T> getVertexAttributeValues(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& attributeName) {
    const auto attribute = primitive.attributes.find(attributeName);
    if (attribute == primitive.attributes.end()) {
        return {};
    }

    auto accessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, attribute->second);
    if (!accessor) {
        return {};
    }

    auto view = CesiumGltf::AccessorView<GetNativeType<T>>(model, *accessor);

    if (view.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return VertexAttributeAccessor<T>(view, accessor->normalized);
}

// Explicit template instantiation
// clang-format off
template VertexAttributeAccessor<VertexAttributeType::UINT8> getVertexAttributeValues<VertexAttributeType::UINT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::INT8> getVertexAttributeValues<VertexAttributeType::INT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::UINT16> getVertexAttributeValues<VertexAttributeType::UINT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::INT16> getVertexAttributeValues<VertexAttributeType::INT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::FLOAT32> getVertexAttributeValues<VertexAttributeType::FLOAT32>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC2_UINT8> getVertexAttributeValues<VertexAttributeType::VEC2_UINT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC2_INT8> getVertexAttributeValues<VertexAttributeType::VEC2_INT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC2_UINT16> getVertexAttributeValues<VertexAttributeType::VEC2_UINT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC2_INT16> getVertexAttributeValues<VertexAttributeType::VEC2_INT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC2_FLOAT32> getVertexAttributeValues<VertexAttributeType::VEC2_FLOAT32>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC3_UINT8> getVertexAttributeValues<VertexAttributeType::VEC3_UINT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC3_INT8> getVertexAttributeValues<VertexAttributeType::VEC3_INT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC3_UINT16> getVertexAttributeValues<VertexAttributeType::VEC3_UINT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC3_INT16> getVertexAttributeValues<VertexAttributeType::VEC3_INT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC3_FLOAT32> getVertexAttributeValues<VertexAttributeType::VEC3_FLOAT32>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC4_UINT8> getVertexAttributeValues<VertexAttributeType::VEC4_UINT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC4_INT8> getVertexAttributeValues<VertexAttributeType::VEC4_INT8>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC4_UINT16> getVertexAttributeValues<VertexAttributeType::VEC4_UINT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC4_INT16> getVertexAttributeValues<VertexAttributeType::VEC4_INT16>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
template VertexAttributeAccessor<VertexAttributeType::VEC4_FLOAT32> getVertexAttributeValues<VertexAttributeType::VEC4_FLOAT32>(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, const std::string& attributeName);
// clang-format on

const CesiumGltf::ImageCesium*
getBaseColorTextureImage(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    if (!hasMaterial(primitive)) {
        return nullptr;
    }

    const auto& material = model.materials[static_cast<size_t>(primitive.material)];

    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value() && pbrMetallicRoughness->baseColorTexture.has_value()) {
        const auto index = pbrMetallicRoughness->baseColorTexture.value().index;
        if (index >= 0 && static_cast<size_t>(index) < model.textures.size()) {
            const auto& texture = model.textures[static_cast<size_t>(index)];
            return &getImageCesium(model, texture);
        }
    }

    return nullptr;
}

const CesiumGltf::ImageCesium* getFeatureIdTextureImage(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t featureIdSetIndex) {
    if (!primitive.hasExtension<CesiumGltf::ExtensionExtMeshFeatures>()) {
        return nullptr;
    }

    const auto& extMeshFeatures = *primitive.getExtension<CesiumGltf::ExtensionExtMeshFeatures>();
    if (featureIdSetIndex >= extMeshFeatures.featureIds.size()) {
        return nullptr;
    }

    const auto& featureId = extMeshFeatures.featureIds[featureIdSetIndex];
    if (!featureId.texture.has_value()) {
        return nullptr;
    }

    const auto index = featureId.texture.value().index;
    if (index < 0 || static_cast<size_t>(index) >= model.textures.size()) {
        return nullptr;
    }

    const auto& texture = model.textures[static_cast<size_t>(index)];

    return &getImageCesium(model, texture);
}

MaterialInfo getMaterialInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    if (!hasMaterial(primitive)) {
        return getDefaultMaterialInfo();
    }

// Ignore uninitialized member warning from gcc 11.2.0. This warning is not reported in gcc 11.4.0
#ifdef CESIUM_OMNI_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    auto materialInfo = getDefaultMaterialInfo();
#pragma GCC diagnostic pop
#endif

    const auto& material = model.materials[static_cast<size_t>(primitive.material)];

    materialInfo.alphaCutoff = getAlphaCutoff(material);
    materialInfo.alphaMode = getAlphaMode(material);
    materialInfo.emissiveFactor = getEmissiveFactor(material);
    materialInfo.doubleSided = getDoubleSided(material);

    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        materialInfo.baseAlpha = getBaseAlpha(pbrMetallicRoughness.value());
        materialInfo.baseColorFactor = getBaseColorFactor(pbrMetallicRoughness.value());
        materialInfo.metallicFactor = getMetallicFactor(pbrMetallicRoughness.value());
        materialInfo.roughnessFactor = getRoughnessFactor(pbrMetallicRoughness.value());

        if (pbrMetallicRoughness.value().baseColorTexture.has_value()) {
            materialInfo.baseColorTexture =
                getTextureInfo(model, pbrMetallicRoughness.value().baseColorTexture.value());
        }
    }

    if (material.hasExtension<CesiumGltf::ExtensionKhrMaterialsUnlit>()) {
        // Unlit materials aren't supported in Omniverse yet but we can hard code the material values to something reasonable
        materialInfo.metallicFactor = 0.0;
        materialInfo.roughnessFactor = 1.0;
    }

    materialInfo.hasVertexColors = hasVertexColors(model, primitive, 0);

    return materialInfo;
}

FeaturesInfo getFeaturesInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    if (!primitive.hasExtension<CesiumGltf::ExtensionExtMeshFeatures>()) {
        return {};
    }

    const auto& extMeshFeatures = *primitive.getExtension<CesiumGltf::ExtensionExtMeshFeatures>();
    const auto& featureIds = extMeshFeatures.featureIds;

    FeaturesInfo featuresInfo;
    featuresInfo.featureIds.reserve(featureIds.size());

    for (const auto& featureId : featureIds) {
        const auto nullFeatureId = castOptional<uint64_t>(featureId.nullFeatureId);
        const auto featureCount = static_cast<uint64_t>(featureId.featureCount);

        auto featureIdStorage = std::variant<std::monostate, uint64_t, TextureInfo>();

        if (featureId.attribute.has_value()) {
            featureIdStorage = static_cast<uint64_t>(featureId.attribute.value());
        } else if (featureId.texture.has_value()) {
            featureIdStorage = getFeatureIdTextureInfo(model, featureId.texture.value());
        } else {
            featureIdStorage = std::monostate();
        }

        featuresInfo.featureIds.emplace_back(FeatureId{nullFeatureId, featureCount, featureIdStorage});
    }

    return featuresInfo;
}

std::set<VertexAttributeInfo>
getCustomVertexAttributes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    constexpr std::array<const char*, 8> knownSemantics = {{
        "POSITION",
        "NORMAL",
        "TANGENT",
        "TEXCOORD",
        "COLOR",
        "JOINTS",
        "WEIGHTS",
        "_CESIUMOVERLAY",
    }};

    std::set<VertexAttributeInfo> customVertexAttributes;

    for (const auto& attribute : primitive.attributes) {
        const auto& attributeName = attribute.first;
        const auto [semantic, setIndex] = parseAttributeName(attributeName);
        if (std::find(knownSemantics.begin(), knownSemantics.end(), semantic) != knownSemantics.end()) {
            continue;
        }

        auto accessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, static_cast<int32_t>(attribute.second));
        if (!accessor) {
            continue;
        }

        const auto valid = createAccessorView(model, *accessor, [](const auto& accessorView) {
            return accessorView.status() == CesiumGltf::AccessorViewStatus::Valid;
        });

        if (!valid) {
            continue;
        }

        const auto type = getVertexAttributeTypeFromGltf(accessor->type, accessor->componentType);

        if (!type.has_value()) {
            continue;
        }

        const auto fabricAttributeNameStr = fmt::format("primvars:{}", attributeName);
        const auto fabricAttributeName = omni::fabric::Token(fabricAttributeNameStr.c_str());

        customVertexAttributes.insert(VertexAttributeInfo{
            type.value(),
            fabricAttributeName,
            attributeName,
        });
    }

    return customVertexAttributes;
}

const MaterialInfo& getDefaultMaterialInfo() {
    static const auto defaultInfo = MaterialInfo{
        getDefaultAlphaCutoff(),
        getDefaultAlphaMode(),
        getDefaultBaseAlpha(),
        getDefaultBaseColorFactor(),
        getDefaultEmissiveFactor(),
        getDefaultMetallicFactor(),
        getDefaultRoughnessFactor(),
        getDefaultDoubleSided(),
        false,
        std::nullopt,
    };

    return defaultInfo;
}

const TextureInfo& getDefaultTextureInfo() {
    static const auto defaultInfo = TextureInfo{
        getDefaultTexcoordOffset(),
        getDefaultTexcoordRotation(),
        getDefaultTexcoordScale(),
        getDefaultTexcoordSetIndex(),
        getDefaultWrapS(),
        getDefaultWrapT(),
        true,
        {},
    };

    return defaultInfo;
}

bool hasNormals(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, bool smoothNormals) {
    return smoothNormals || getNormalsView(model, primitive).status() == CesiumGltf::AccessorViewStatus::Valid;
}

bool hasTexcoords(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getTexcoordsView(model, primitive, "TEXCOORD", setIndex).status() == CesiumGltf::AccessorViewStatus::Valid;
}

bool hasImageryTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex) {
    return getTexcoordsView(model, primitive, "_CESIUMOVERLAY", setIndex).status() ==
           CesiumGltf::AccessorViewStatus::Valid;
}

std::vector<uint64_t>
getTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    auto setIndexes = std::vector<uint64_t>();

    for (const auto& attribute : primitive.attributes) {
        const auto [semantic, setIndex] = parseAttributeName(attribute.first);
        if (semantic == "TEXCOORD") {
            if (hasTexcoords(model, primitive, setIndex)) {
                setIndexes.push_back(setIndex);
            }
        }
    }

    return setIndexes;
}

std::vector<uint64_t>
getImageryTexcoordSetIndexes(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    auto setIndexes = std::vector<uint64_t>();

    for (const auto& attribute : primitive.attributes) {
        const auto [semantic, setIndex] = parseAttributeName(attribute.first);
        if (semantic == "_CESIUMOVERLAY") {
            if (hasImageryTexcoords(model, primitive, setIndex)) {
                setIndexes.push_back(setIndex);
            }
        }
    }

    return setIndexes;
}

bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getVertexColors(model, primitive, setIndex).size() > 0;
}

bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive) {
    return primitive.material >= 0;
}

} // namespace cesium::omniverse::GltfUtil

namespace cesium::omniverse {

FeatureIdType getFeatureIdType(const FeatureId& featureId) {
    if (std::holds_alternative<std::monostate>(featureId.featureIdStorage)) {
        return FeatureIdType::INDEX;
    } else if (std::holds_alternative<uint64_t>(featureId.featureIdStorage)) {
        return FeatureIdType::ATTRIBUTE;
    } else if (std::holds_alternative<TextureInfo>(featureId.featureIdStorage)) {
        return FeatureIdType::TEXTURE;
    }

    assert(false);
    return FeatureIdType::INDEX;
}

std::vector<FeatureIdType> getFeatureIdTypes(const FeaturesInfo& featuresInfo) {
    const auto& featureIds = featuresInfo.featureIds;

    std::vector<FeatureIdType> featureIdTypes;
    featureIdTypes.reserve(featureIds.size());

    for (const auto& featureId : featureIds) {
        featureIdTypes.push_back(getFeatureIdType(featureId));
    }

    return featureIdTypes;
}

std::vector<uint64_t> getSetIndexMapping(const FeaturesInfo& featuresInfo, FeatureIdType type) {
    const auto& featureIds = featuresInfo.featureIds;

    std::vector<uint64_t> setIndexMapping;
    setIndexMapping.reserve(featureIds.size());

    for (uint64_t i = 0; i < featureIds.size(); i++) {
        if (getFeatureIdType(featureIds[i]) == type) {
            setIndexMapping.push_back(i);
        }
    }

    return setIndexMapping;
}

bool hasFeatureIdType(const FeaturesInfo& featuresInfo, FeatureIdType type) {
    const auto& featureIds = featuresInfo.featureIds;

    for (const auto& featureId : featureIds) {
        if (getFeatureIdType(featureId) == type) {
            return true;
        }
    }

    return false;
}

// In C++ 20 we can use the default equality comparison (= default)
bool TextureInfo::operator==(const TextureInfo& other) const {
    return offset == other.offset && rotation == other.rotation && scale == other.scale && setIndex == other.setIndex &&
           wrapS == other.wrapS && wrapT == other.wrapT && flipVertical == other.flipVertical;
}

// In C++ 20 we can use the default equality comparison (= default)
bool MaterialInfo::operator==(const MaterialInfo& other) const {
    return alphaCutoff == other.alphaCutoff && alphaMode == other.alphaMode && baseAlpha == other.baseAlpha &&
           baseColorFactor == other.baseColorFactor && emissiveFactor == other.emissiveFactor &&
           metallicFactor == other.metallicFactor && roughnessFactor == other.roughnessFactor &&
           doubleSided == other.doubleSided && hasVertexColors == other.hasVertexColors &&
           baseColorTexture == other.baseColorTexture;
}

// In C++ 20 we can use the default equality comparison (= default)
bool VertexAttributeInfo::operator==(const VertexAttributeInfo& other) const {
    return type == other.type && fabricAttributeName == other.fabricAttributeName &&
           gltfAttributeName == other.gltfAttributeName;
}

// This is needed for std::set to be sorted
bool VertexAttributeInfo::operator<(const VertexAttributeInfo& other) const {
    return fabricAttributeName < other.fabricAttributeName;
}

} // namespace cesium::omniverse
