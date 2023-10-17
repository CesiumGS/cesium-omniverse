#include "cesium/omniverse/GltfUtil.h"

#include "cesium/omniverse/LoggerSink.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/ExtensionKhrMaterialsUnlit.h>
#include <CesiumGltf/ExtensionKhrTextureTransform.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltf/TextureInfo.h>
#include <spdlog/fmt/fmt.h>

#include <charconv>
#include <numeric>
#include <regex>

namespace cesium::omniverse::GltfUtil {

namespace {

const CesiumGltf::Material defaultMaterial;
const CesiumGltf::MaterialPBRMetallicRoughness defaultPbrMetallicRoughness;
const CesiumGltf::Sampler defaultSampler;
const CesiumGltf::ExtensionKhrTextureTransform defaultTextureTransform;
const CesiumGltf::TextureInfo defaultTextureInfo;

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

int32_t getAlphaMode(const CesiumGltf::Material& material) {
    if (material.alphaMode == CesiumGltf::Material::AlphaMode::OPAQUE) {
        return 0;
    } else if (material.alphaMode == CesiumGltf::Material::AlphaMode::MASK) {
        return 1;
    } else if (material.alphaMode == CesiumGltf::Material::AlphaMode::BLEND) {
        return 2;
    }

    return 0;
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

int32_t getDefaultAlphaMode() {
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

    textureInfo.setIndex = textureInfoGltf.texCoord;

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
            const auto& sampler = model.samplers[samplerIndex];
            textureInfo.wrapS = getWrapS(sampler);
            textureInfo.wrapT = getWrapT(sampler);
        }
    }

    return textureInfo;
}

const CesiumGltf::ImageCesium& getImageCesium(const CesiumGltf::Model& model, const CesiumGltf::Texture& texture) {
    const auto imageId = static_cast<uint64_t>(texture.source);
    const auto& image = model.images[imageId];
    return image.cesium;
}

std::pair<std::string, uint64_t> parseAttributeName(const std::string& attributeName) {
    const auto regex = std::regex("^([a-zA-Z_]*)(_([1-9]\\d*|0))?$");

    std::string semantic;
    std::string setIndex;

    std::smatch matches;

    if (std::regex_match(attributeName, matches, regex)) {
        semantic = matches[1];
        setIndex = matches[3];
    }

    uint64_t setIndexU64 = 0;
    std::from_chars(setIndex.data(), setIndex.data() + setIndex.size(), setIndexU64);

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

MaterialInfo getMaterialInfo(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    if (!hasMaterial(primitive)) {
        return getDefaultMaterialInfo();
    }

    auto materialInfo = getDefaultMaterialInfo();

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

MaterialInfo getDefaultMaterialInfo() {
    return {
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
}

TextureInfo getDefaultTextureInfo() {
    return {
        getDefaultTexcoordOffset(),
        getDefaultTexcoordRotation(),
        getDefaultTexcoordScale(),
        getDefaultTexcoordSetIndex(),
        getDefaultWrapS(),
        getDefaultWrapT(),
        true,
    };
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

// In C++ 20 we can use the default equality comparison (= default)
bool TextureInfo::operator==(const TextureInfo& other) const {
    if (offset != other.offset) {
        return false;
    }

    if (rotation != other.rotation) {
        return false;
    }

    if (scale != other.scale) {
        return false;
    }

    if (setIndex != other.setIndex) {
        return false;
    }

    if (wrapS != other.wrapS) {
        return false;
    }

    if (wrapT != other.wrapT) {
        return false;
    }

    if (flipVertical != other.flipVertical) {
        return false;
    }
    return true;
}

// In C++ 20 we can use the default equality comparison (= default)
bool MaterialInfo::operator==(const MaterialInfo& other) const {
    if (alphaCutoff != other.alphaCutoff) {
        return false;
    }

    if (alphaMode != other.alphaMode) {
        return false;
    }

    if (baseAlpha != other.baseAlpha) {
        return false;
    }

    if (baseColorFactor != other.baseColorFactor) {
        return false;
    }

    if (emissiveFactor != other.emissiveFactor) {
        return false;
    }

    if (metallicFactor != other.metallicFactor) {
        return false;
    }

    if (roughnessFactor != other.roughnessFactor) {
        return false;
    }

    if (doubleSided != other.doubleSided) {
        return false;
    }

    if (hasVertexColors != other.hasVertexColors) {
        return false;
    }

    // != operator doesn't compile for some reason
    if (!(baseColorTexture == other.baseColorTexture)) {
        return false;
    }

    return true;
}

} // namespace cesium::omniverse
