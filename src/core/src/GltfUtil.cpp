#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/ExtensionKhrMaterialsUnlit.h>
#include <CesiumGltf/Model.h>
#include <pxr/base/gf/range3d.h>
#include <spdlog/fmt/fmt.h>

#include <numeric>

namespace cesium::omniverse::GltfUtil {

namespace {

const CesiumGltf::Material defaultMaterial;
const CesiumGltf::MaterialPBRMetallicRoughness defaultPbrMetallicRoughness;
const CesiumGltf::Sampler defaultSampler;

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
    const glm::fvec2& translation,
    const glm::fvec2& scale,
    bool flipVertical) {

    const auto texcoordsView = getTexcoordsView(model, primitive, semantic, setIndex);

    if (texcoordsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return TexcoordsAccessor(texcoordsView, translation, scale, flipVertical);
}

template <typename VertexColorType>
VertexColorsAccessor getVertexColorsAccessor(const CesiumGltf::Model& model, const CesiumGltf::Accessor& accessor) {
    CesiumGltf::AccessorView<VertexColorType> view{model, accessor};
    if (view.status() == CesiumGltf::AccessorViewStatus::Valid) {
        return VertexColorsAccessor(view);
    }
    return {};
}

float getBaseAlpha(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return static_cast<float>(pbrMetallicRoughness.baseColorFactor[3]);
}

pxr::GfVec3f getBaseColorFactor(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return pxr::GfVec3f(
        static_cast<float>(pbrMetallicRoughness.baseColorFactor[0]),
        static_cast<float>(pbrMetallicRoughness.baseColorFactor[1]),
        static_cast<float>(pbrMetallicRoughness.baseColorFactor[2]));
}

float getMetallicFactor(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return static_cast<float>(pbrMetallicRoughness.metallicFactor);
}

float getRoughnessFactor(const CesiumGltf::MaterialPBRMetallicRoughness& pbrMetallicRoughness) {
    return static_cast<float>(pbrMetallicRoughness.roughnessFactor);
}

int getWrapS(const CesiumGltf::Sampler& sampler) {
    return sampler.wrapS;
}

int getWrapT(const CesiumGltf::Sampler& sampler) {
    return sampler.wrapT;
}

} // namespace

PositionsAccessor getPositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    const auto positionsView = getPositionsView(model, primitive);

    if (positionsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    return PositionsAccessor(positionsView);
}

std::optional<pxr::GfRange3d> getExtent(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
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

    return pxr::GfRange3d(pxr::GfVec3d(min[0], min[1], min[2]), pxr::GfVec3d(max[0], max[1], max[2]));
}

IndicesAccessor getIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PositionsAccessor& positions) {
    const auto indicesAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, primitive.indices);
    if (!indicesAccessor) {
        return IndicesAccessor(positions.size());
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

NormalsAccessor getNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const PositionsAccessor& positions,
    const IndicesAccessor& indices,
    bool smoothNormals) {

    const auto normalsView = getNormalsView(model, primitive);

    if (normalsView.status() == CesiumGltf::AccessorViewStatus::Valid) {
        return NormalsAccessor(normalsView);
    }

    if (smoothNormals) {
        return NormalsAccessor::GenerateSmooth(positions, indices);
    }

    // Otherwise if normals are missing and smoothNormals is false Omniverse will generate flat normals for us automatically
    return {};
}

TexcoordsAccessor getTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex,
    const glm::fvec2& translation,
    const glm::fvec2& scale) {
    return getTexcoords(model, primitive, "TEXCOORD", setIndex, translation, scale, true);
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

FaceVertexCountsAccessor getFaceVertexCounts(const IndicesAccessor& indices) {
    return FaceVertexCountsAccessor(indices.size() / 3);
}

float getAlphaCutoff(const CesiumGltf::Material& material) {
    return static_cast<float>(material.alphaCutoff);
}

int getAlphaMode(const CesiumGltf::Material& material) {
    if (material.alphaMode == CesiumGltf::Material::AlphaMode::OPAQUE) {
        return 0;
    } else if (material.alphaMode == CesiumGltf::Material::AlphaMode::MASK) {
        return 1;
    } else if (material.alphaMode == CesiumGltf::Material::AlphaMode::BLEND) {
        return 2;
    }

    return 0;
}

float getBaseAlpha(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return getBaseAlpha(pbrMetallicRoughness.value());
    }

    return getDefaultBaseAlpha();
}

pxr::GfVec3f getBaseColorFactor(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return getBaseColorFactor(pbrMetallicRoughness.value());
    }

    return getDefaultBaseColorFactor();
}

pxr::GfVec3f getEmissiveFactor(const CesiumGltf::Material& material) {
    return pxr::GfVec3f(
        static_cast<float>(material.emissiveFactor[0]),
        static_cast<float>(material.emissiveFactor[1]),
        static_cast<float>(material.emissiveFactor[2]));
}

float getMetallicFactor(const CesiumGltf::Material& material) {
    if (material.hasExtension<CesiumGltf::ExtensionKhrMaterialsUnlit>()) {
        // Unlit materials aren't supported in Omniverse yet but we can hard code the metallic factor to something reasonable
        return 0.0f;
    }

    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return getMetallicFactor(pbrMetallicRoughness.value());
    }

    return getDefaultMetallicFactor();
}

float getRoughnessFactor(const CesiumGltf::Material& material) {
    if (material.hasExtension<CesiumGltf::ExtensionKhrMaterialsUnlit>()) {
        // Unlit materials aren't supported in Omniverse yet but we can hard code the roughness factor to something reasonable
        return 1.0f;
    }

    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return getRoughnessFactor(pbrMetallicRoughness.value());
    }

    return getDefaultRoughnessFactor();
}

int getBaseColorTextureWrapS(const CesiumGltf::Model& model, const CesiumGltf::Material& material) {
    const auto baseColorTextureIndex = getBaseColorTextureIndex(model, material);

    if (baseColorTextureIndex.has_value()) {
        const auto samplerIndex = model.textures[baseColorTextureIndex.value()].sampler;
        if (samplerIndex != -1) {
            const auto& sampler = model.samplers[samplerIndex];
            return sampler.wrapS;
        }
    }

    return getDefaultWrapS();
}

int getBaseColorTextureWrapT(const CesiumGltf::Model& model, const CesiumGltf::Material& material) {
    const auto baseColorTextureIndex = getBaseColorTextureIndex(model, material);

    if (baseColorTextureIndex.has_value()) {
        const auto samplerIndex = model.textures[baseColorTextureIndex.value()].sampler;
        if (samplerIndex != -1) {
            const auto& sampler = model.samplers[samplerIndex];
            return sampler.wrapT;
        }
    }

    return getDefaultWrapT();
}

float getDefaultAlphaCutoff() {
    return getAlphaCutoff(defaultMaterial);
}

int getDefaultAlphaMode() {
    return getAlphaMode(defaultMaterial);
}

float getDefaultBaseAlpha() {
    return getBaseAlpha(defaultPbrMetallicRoughness);
}

pxr::GfVec3f getDefaultBaseColorFactor() {
    return getBaseColorFactor(defaultPbrMetallicRoughness);
}

pxr::GfVec3f getDefaultEmissiveFactor() {
    return getEmissiveFactor(defaultMaterial);
}

float getDefaultMetallicFactor() {
    return getMetallicFactor(defaultPbrMetallicRoughness);
}

float getDefaultRoughnessFactor() {
    return getRoughnessFactor(defaultPbrMetallicRoughness);
}

int getDefaultWrapS() {
    return getWrapS(defaultSampler);
}

int getDefaultWrapT() {
    return getWrapT(defaultSampler);
}

std::optional<uint64_t> getBaseColorTextureIndex(const CesiumGltf::Model& model, const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value() && pbrMetallicRoughness->baseColorTexture.has_value()) {
        const auto index = pbrMetallicRoughness->baseColorTexture->index;
        if (index >= 0 && static_cast<size_t>(index) < model.textures.size()) {
            return static_cast<uint64_t>(index);
        }
    }

    return std::nullopt;
}

bool getDoubleSided(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    if (primitive.material == -1) {
        return false;
    }

    const auto& material = model.materials[static_cast<size_t>(primitive.material)];
    return material.doubleSided;
}

TexcoordsAccessor getImageryTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex,
    const glm::fvec2& translation,
    const glm::fvec2& scale) {
    return getTexcoords(model, primitive, "_CESIUMOVERLAY", setIndex, translation, scale, false);
}

const CesiumGltf::ImageCesium& getImageCesium(const CesiumGltf::Model& model, const CesiumGltf::Texture& texture) {
    const auto imageId = static_cast<uint64_t>(texture.source);
    const auto& image = model.images[imageId];
    return image.cesium;
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

bool hasVertexColors(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getVertexColors(model, primitive, setIndex).size() > 0;
}

bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive) {
    return primitive.material >= 0;
}

} // namespace cesium::omniverse::GltfUtil
