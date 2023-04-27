#include "cesium/omniverse/GltfUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/Model.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <spdlog/fmt/fmt.h>

#include <numeric>

namespace cesium::omniverse::GltfUtil {

namespace {

const CesiumGltf::MaterialPBRMetallicRoughness defaultPbrMetallicRoughness;

template <typename IndexType>
pxr::VtArray<int> createIndices(
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::AccessorView<IndexType>& indicesAccessorView) {
    if (indicesAccessorView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLES) {
        if (indicesAccessorView.size() % 3 != 0) {
            return {};
        }

        pxr::VtArray<int> indices;
        indices.reserve(static_cast<uint64_t>(indicesAccessorView.size()));
        for (auto i = 0; i < indicesAccessorView.size(); i++) {
            indices.push_back(static_cast<int>(indicesAccessorView[i]));
        }

        return indices;
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLE_STRIP) {
        if (indicesAccessorView.size() <= 2) {
            return {};
        }

        pxr::VtArray<int> indices;
        indices.reserve(static_cast<uint64_t>(indicesAccessorView.size() - 2) * 3);
        for (auto i = 0; i < indicesAccessorView.size() - 2; i++) {
            if (i % 2) {
                indices.push_back(static_cast<int>(indicesAccessorView[i]));
                indices.push_back(static_cast<int>(indicesAccessorView[i + 2]));
                indices.push_back(static_cast<int>(indicesAccessorView[i + 1]));
            } else {
                indices.push_back(static_cast<int>(indicesAccessorView[i]));
                indices.push_back(static_cast<int>(indicesAccessorView[i + 1]));
                indices.push_back(static_cast<int>(indicesAccessorView[i + 2]));
            }
        }

        return indices;
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLE_FAN) {
        if (indicesAccessorView.size() <= 2) {
            return {};
        }

        pxr::VtArray<int> indices;
        indices.reserve(static_cast<uint64_t>(indicesAccessorView.size() - 2) * 3);
        for (auto i = 0; i < indicesAccessorView.size() - 2; i++) {
            indices.push_back(static_cast<int>(indicesAccessorView[0]));
            indices.push_back(static_cast<int>(indicesAccessorView[i + 1]));
            indices.push_back(static_cast<int>(indicesAccessorView[i + 2]));
        }

        return indices;
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

pxr::VtArray<pxr::GfVec2f> getTexcoords(
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

    pxr::VtArray<pxr::GfVec2f> usdTexcoords;
    usdTexcoords.reserve(static_cast<size_t>(texcoordsView.size()));

    const auto applyTransform = translation != glm::fvec2(0.0, 0.0) && scale != glm::fvec2(1.0, 1.0);

    for (auto i = 0; i < texcoordsView.size(); ++i) {
        auto texcoord = texcoordsView[i];

        if (flipVertical) {
            texcoord.y = 1.0f - texcoord.y;
        }

        if (applyTransform) {
            texcoord = texcoord * scale + translation;
        }

        usdTexcoords.push_back(pxr::GfVec2f(texcoord.x, texcoord.y));
    }

    return usdTexcoords;
}

} // namespace

pxr::VtArray<pxr::GfVec3f> getPositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    const auto positionsView = getPositionsView(model, primitive);

    if (positionsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    pxr::VtArray<pxr::GfVec3f> usdPositions;
    usdPositions.reserve(static_cast<size_t>(positionsView.size()));

    for (auto i = 0; i < positionsView.size(); i++) {
        const auto& position = positionsView[i];
        usdPositions.push_back(pxr::GfVec3f(position.x, position.y, position.z));
    }

    return usdPositions;
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

pxr::VtArray<int> getIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions) {
    const auto indicesAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, primitive.indices);
    if (!indicesAccessor) {
        pxr::VtArray<int> indices(positions.size());
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    if (indicesAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_BYTE) {
        CesiumGltf::AccessorView<std::uint8_t> view{model, *indicesAccessor};
        return createIndices(primitive, view);
    } else if (indicesAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_SHORT) {
        CesiumGltf::AccessorView<std::uint16_t> view{model, *indicesAccessor};
        return createIndices(primitive, view);
    } else if (indicesAccessor->componentType == CesiumGltf::AccessorSpec::ComponentType::UNSIGNED_INT) {
        CesiumGltf::AccessorView<std::uint32_t> view{model, *indicesAccessor};
        return createIndices(primitive, view);
    }

    return {};
}

pxr::VtArray<pxr::GfVec3f> getNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions,
    const pxr::VtArray<int>& indices,
    bool smoothNormals) {

    const auto normalsView = getNormalsView(model, primitive);

    if (normalsView.status() == CesiumGltf::AccessorViewStatus::Valid) {
        pxr::VtArray<pxr::GfVec3f> normalsUsd;
        normalsUsd.reserve(static_cast<size_t>(normalsView.size()));
        for (auto i = 0; i < normalsView.size(); ++i) {
            const auto& normal = normalsView[i];
            normalsUsd.push_back(pxr::GfVec3f(normal.x, normal.y, normal.z));
        }
        return normalsUsd;
    }

    if (smoothNormals) {
        pxr::VtArray<pxr::GfVec3f> normalsUsd(positions.size(), pxr::GfVec3f(0.0f));

        for (size_t i = 0; i < indices.size(); i += 3) {
            auto idx0 = static_cast<size_t>(indices[i]);
            auto idx1 = static_cast<size_t>(indices[i + 1]);
            auto idx2 = static_cast<size_t>(indices[i + 2]);

            const auto& p0 = positions[idx0];
            const auto& p1 = positions[idx1];
            const auto& p2 = positions[idx2];
            auto n = pxr::GfCross(p1 - p0, p2 - p0);
            n.Normalize();

            normalsUsd[idx0] += n;
            normalsUsd[idx1] += n;
            normalsUsd[idx2] += n;
        }

        for (auto& n : normalsUsd) {
            n.Normalize();
        }

        return normalsUsd;
    }

    // Otherwise if normals are missing and smoothNormals is false Omniverse will generate flat normals for us automatically
    return {};
}

pxr::VtArray<pxr::GfVec2f> getTexcoords(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    uint64_t setIndex,
    const glm::fvec2& translation,
    const glm::fvec2& scale) {
    return getTexcoords(model, primitive, "TEXCOORD", setIndex, translation, scale, true);
}

pxr::VtArray<int> getFaceVertexCounts(const pxr::VtArray<int>& indices) {
    pxr::VtArray<int> faceVertexCounts(indices.size() / 3, 3);
    return faceVertexCounts;
}

pxr::GfVec3f getBaseColorFactor(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return pxr::GfVec3f(
            static_cast<float>(pbrMetallicRoughness.value().baseColorFactor[0]),
            static_cast<float>(pbrMetallicRoughness.value().baseColorFactor[1]),
            static_cast<float>(pbrMetallicRoughness.value().baseColorFactor[2]));
    }

    return getDefaultBaseColorFactor();
}

float getMetallicFactor(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return static_cast<float>(pbrMetallicRoughness->metallicFactor);
    }

    return getDefaultMetallicFactor();
}

float getRoughnessFactor(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return static_cast<float>(pbrMetallicRoughness->roughnessFactor);
    }

    return getDefaultRoughnessFactor();
}

pxr::GfVec3f getDefaultBaseColorFactor() {
    return pxr::GfVec3f(
        static_cast<float>(defaultPbrMetallicRoughness.baseColorFactor[0]),
        static_cast<float>(defaultPbrMetallicRoughness.baseColorFactor[1]),
        static_cast<float>(defaultPbrMetallicRoughness.baseColorFactor[2]));
}

float getDefaultMetallicFactor() {
    return static_cast<float>(defaultPbrMetallicRoughness.metallicFactor);
}

float getDefaultRoughnessFactor() {
    return static_cast<float>(defaultPbrMetallicRoughness.roughnessFactor);
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

pxr::VtArray<pxr::GfVec2f> getImageryTexcoords(
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

bool hasMaterial(const CesiumGltf::MeshPrimitive& primitive) {
    return primitive.material >= 0;
}

} // namespace cesium::omniverse::GltfUtil
