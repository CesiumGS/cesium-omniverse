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

pxr::VtArray<pxr::GfVec2f> getUVs(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& semantic,
    uint64_t setIndex,
    bool flipUVs) {

    const auto uvAttribute = primitive.attributes.find(fmt::format("{}_{}", semantic, setIndex));
    if (uvAttribute == primitive.attributes.end()) {
        return {};
    }

    auto uvAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, uvAttribute->second);
    if (!uvAccessor) {
        return {};
    }

    auto uvsView = CesiumGltf::AccessorView<glm::fvec2>(model, *uvAccessor);

    if (uvsView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    pxr::VtArray<pxr::GfVec2f> usdUVs;
    usdUVs.reserve(static_cast<size_t>(uvsView.size()));

    for (auto i = 0; i < uvsView.size(); ++i) {
        auto uv = uvsView[i];

        if (flipUVs) {
            uv.y = 1.0f - uv.y;
        }

        usdUVs.push_back(pxr::GfVec2f(uv.x, uv.y));
    }

    return usdUVs;
}

} // namespace

pxr::VtArray<pxr::GfVec3f>
getPrimitivePositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    auto positionAttribute = primitive.attributes.find("POSITION");
    if (positionAttribute == primitive.attributes.end()) {
        return {};
    }

    auto positionAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, positionAttribute->second);
    if (!positionAccessor) {
        return {};
    }

    auto positionView = CesiumGltf::AccessorView<glm::fvec3>(model, *positionAccessor);
    if (positionView.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    pxr::VtArray<pxr::GfVec3f> usdPositions;
    usdPositions.reserve(static_cast<size_t>(positionView.size()));

    for (auto i = 0; i < positionView.size(); i++) {
        const auto& position = positionView[i];
        usdPositions.push_back(pxr::GfVec3f(position.x, position.y, position.z));
    }

    return usdPositions;
}

std::optional<pxr::GfRange3d>
getPrimitiveExtent(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
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

pxr::VtArray<int> getPrimitiveIndices(
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

pxr::VtArray<pxr::GfVec3f> getPrimitiveNormals(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions,
    const pxr::VtArray<int>& indices) {
    auto normalAttribute = primitive.attributes.find("NORMAL");
    if (normalAttribute != primitive.attributes.end()) {
        const auto normalsView = CesiumGltf::AccessorView<glm::fvec3>(model, normalAttribute->second);
        if (normalsView.status() == CesiumGltf::AccessorViewStatus::Valid) {
            pxr::VtArray<pxr::GfVec3f> normalsUsd;
            normalsUsd.reserve(static_cast<size_t>(normalsView.size()));
            for (auto i = 0; i < normalsView.size(); ++i) {
                const auto& normal = normalsView[i];
                normalsUsd.push_back(pxr::GfVec3f(normal.x, normal.y, normal.z));
            }
            return normalsUsd;
        }
    }

    // Generate smooth normals
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

pxr::VtArray<pxr::GfVec2f>
getPrimitiveUVs(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getUVs(model, primitive, "TEXCOORD", setIndex, true);
}

pxr::VtArray<int> getPrimitiveFaceVertexCounts(const pxr::VtArray<int>& indices) {
    pxr::VtArray<int> faceVertexCounts(indices.size() / 3, 3);
    return faceVertexCounts;
}

pxr::GfVec3f getBaseColorFactor(const CesiumGltf::Material& material) {
    pxr::GfVec3f baseColorFactor(1.0, 1.0, 1.0);

    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        baseColorFactor[0] = static_cast<float>(pbrMetallicRoughness->baseColorFactor[0]);
        baseColorFactor[1] = static_cast<float>(pbrMetallicRoughness->baseColorFactor[1]);
        baseColorFactor[2] = static_cast<float>(pbrMetallicRoughness->baseColorFactor[2]);
    }

    return baseColorFactor;
}

float getMetallicFactor(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return static_cast<float>(pbrMetallicRoughness->metallicFactor);
    }

    return 0.0f;
}

float getRoughnessFactor(const CesiumGltf::Material& material) {
    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    if (pbrMetallicRoughness.has_value()) {
        return static_cast<float>(pbrMetallicRoughness->roughnessFactor);
    }

    return 1.0f;
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

pxr::VtArray<pxr::GfVec2f>
getImageryUVs(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive, uint64_t setIndex) {
    return getUVs(model, primitive, "_CESIUMOVERLAY", setIndex, false);
}

const CesiumGltf::ImageCesium& getImageCesium(const CesiumGltf::Model& model, const CesiumGltf::Texture& texture) {
    const auto imageId = static_cast<uint64_t>(texture.source);
    const auto& image = model.images[imageId];
    return image.cesium;
}

} // namespace cesium::omniverse::GltfUtil
