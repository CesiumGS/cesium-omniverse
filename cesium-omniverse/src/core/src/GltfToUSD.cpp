#include "cesium/omniverse/GltfToUSD.h"

#include "cesium/omniverse/InMemoryAssetResolver.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/AccessorView.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltfReader/GltfReader.h>
#include <glm/glm.hpp>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <spdlog/fmt/fmt.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>

static std::string errorMessage;

// clang-format off
namespace pxr {
TF_DEFINE_PRIVATE_TOKENS(
    _tokens,

    // Tokens used for USD Preview Surface
    // Notes below copied from helloWorld.cpp in Connect Sample 200.0.0
    //
    // Private tokens for building up SdfPaths. We recommend
    // constructing SdfPaths via tokens, as there is a performance
    // cost to constructing them directly via strings (effectively,
    // a table lookup per path element). Similarly, any API which
    // takes a token as input should use a predefined token
    // rather than one created on the fly from a string.
    (vertex)
    (diffuseColor)
    (roughness)
    (metallic)
    (normal)
    (file)
    (result)
    (varname)
    (rgb)
    (RAW)
    (sRGB)
    (surface)
    (st)
    (st_0)
    (st_1)
    (wrapS)
    (wrapT)
    (clamp)
    (UsdPreviewSurface)
    ((stPrimvarName, "frame:stPrimvarName"))
    ((UsdShaderId, "UsdPreviewSurface"))
    ((PrimStShaderId, "UsdPrimvarReader_float2"))
    (UsdUVTexture));
}
// clang-format on

namespace Cesium {

namespace {
bool isIdentityMatrix(const std::vector<double>& matrix) {
    static constexpr double identity[] = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    return std::equal(matrix.begin(), matrix.end(), identity);
}

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
        indices.resize(static_cast<std::size_t>(indicesAccessorView.size()));
        for (std::int64_t i = 0; i < indicesAccessorView.size(); ++i) {
            indices[static_cast<std::size_t>(i)] = static_cast<int>(indicesAccessorView[i]);
        }

        return indices;
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLE_STRIP) {
        if (indicesAccessorView.size() <= 2) {
            return {};
        }

        pxr::VtArray<int> indices;
        indices.reserve(static_cast<std::size_t>(indicesAccessorView.size() - 2) * 3);
        for (std::int64_t i = 0; i < indicesAccessorView.size() - 2; ++i) {
            if (i % 2) {
                indices.push_back(static_cast<const int>(indicesAccessorView[i]));
                indices.push_back(static_cast<const int>(indicesAccessorView[i + 2]));
                indices.push_back(static_cast<const int>(indicesAccessorView[i + 1]));
            } else {
                indices.push_back(static_cast<const int>(indicesAccessorView[i]));
                indices.push_back(static_cast<const int>(indicesAccessorView[i + 1]));
                indices.push_back(static_cast<const int>(indicesAccessorView[i + 2]));
            }
        }

        return indices;
    }

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::TRIANGLE_FAN) {
        if (indicesAccessorView.size() <= 2) {
            return {};
        }

        pxr::VtArray<int> indices;
        indices.reserve(static_cast<std::size_t>(indicesAccessorView.size() - 2) * 3);
        for (std::int64_t i = 0; i < indicesAccessorView.size() - 2; ++i) {
            indices.push_back(static_cast<const int>(indicesAccessorView[0]));
            indices.push_back(static_cast<const int>(indicesAccessorView[i + 1]));
            indices.push_back(static_cast<const int>(indicesAccessorView[i + 2]));
        }

        return indices;
    }

    return {};
}

CesiumGltf::AccessorView<glm::vec2> getUVsAccessorView(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& semantic,
    std::int32_t textureCoordId) {
    auto uvAttribute = primitive.attributes.find(fmt::format("{}_{}", semantic, textureCoordId));
    if (uvAttribute == primitive.attributes.end()) {
        return CesiumGltf::AccessorView<glm::vec2>();
    }

    auto uvAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, uvAttribute->second);
    if (!uvAccessor) {
        return CesiumGltf::AccessorView<glm::vec2>();
    }

    return CesiumGltf::AccessorView<glm::vec2>(model, *uvAccessor);
}

pxr::VtArray<pxr::GfVec2f> getUVs(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::string& semantic,
    std::int32_t textureCoordId,
    bool flipUVs) {

    auto uvs = getUVsAccessorView(model, primitive, semantic, textureCoordId);
    if (uvs.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    pxr::VtArray<pxr::GfVec2f> usdUVs;
    usdUVs.reserve(static_cast<std::size_t>(uvs.size()));
    for (int64_t i = 0; i < uvs.size(); ++i) {
        glm::vec2 vert = uvs[i];

        if (flipUVs) {
            vert.y = 1.0f - vert.y;
        }

        usdUVs.push_back(pxr::GfVec2f(vert.x, vert.y));
    }

    return usdUVs;
}

pxr::VtArray<pxr::GfVec2f> getPrimitiveUVs(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    std::int32_t textureCoordId) {
    return getUVs(model, primitive, "TEXCOORD", textureCoordId, true);
}

pxr::VtArray<pxr::GfVec2f> getRasterOverlayUVs(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    std::int32_t rasterOverlayId) {
    return getUVs(model, primitive, "_CESIUMOVERLAY", rasterOverlayId, false);
}

bool hasRasterOverlayUVs(const CesiumGltf::Model& model, std::int32_t rasterOverlayId) {
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            const auto uvs = getUVsAccessorView(model, primitive, "_CESIUMOVERLAY", rasterOverlayId);
            if (uvs.status() == CesiumGltf::AccessorViewStatus::Valid) {
                return true;
            }
        }
    }

    return false;
}

pxr::VtArray<pxr::GfVec3f>
getPrimitivePositions(const CesiumGltf::Model& model, const CesiumGltf::MeshPrimitive& primitive) {
    // retrieve required positions first
    auto positionAttribute = primitive.attributes.find("POSITION");
    if (positionAttribute == primitive.attributes.end()) {
        return {};
    }

    auto positionAccessor = model.getSafe<CesiumGltf::Accessor>(&model.accessors, positionAttribute->second);
    if (!positionAccessor) {
        return {};
    }

    auto positions = CesiumGltf::AccessorView<glm::vec3>(model, *positionAccessor);
    if (positions.status() != CesiumGltf::AccessorViewStatus::Valid) {
        return {};
    }

    pxr::VtArray<pxr::GfVec3f> usdPositions;
    usdPositions.reserve(static_cast<std::size_t>(positions.size()));
    for (int64_t i = 0; i < positions.size(); ++i) {
        const glm::vec3& vert = positions[i];
        usdPositions.push_back(pxr::GfVec3f(vert.x, vert.y, vert.z));
    }

    return usdPositions;
}

pxr::VtArray<int> getPrimitiveIndices(
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const pxr::VtArray<pxr::GfVec3f>& positions) {
    const CesiumGltf::Accessor* indicesAccessor =
        model.getSafe<CesiumGltf::Accessor>(&model.accessors, primitive.indices);
    if (!indicesAccessor) {
        pxr::VtArray<int> indices;
        indices.resize(positions.size());
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
    // get normal view
    auto normalAttribute = primitive.attributes.find("NORMAL");
    if (normalAttribute != primitive.attributes.end()) {
        auto normalsView = CesiumGltf::AccessorView<glm::vec3>(model, normalAttribute->second);
        if (normalsView.status() == CesiumGltf::AccessorViewStatus::Valid) {
            pxr::VtArray<pxr::GfVec3f> normalsUsd;
            normalsUsd.reserve(static_cast<std::size_t>(normalsView.size()));
            for (int64_t i = 0; i < normalsView.size(); ++i) {
                const glm::vec3& n = normalsView[i];
                normalsUsd.push_back(pxr::GfVec3f(n.x, n.y, n.z));
            }

            return normalsUsd;
        }
    }

    // generate normals
    pxr::VtArray<pxr::GfVec3f> normalsUsd(positions.size(), pxr::GfVec3f(0.0f));
    for (std::size_t i = 0; i < indices.size(); i += 3) {
        auto idx0 = static_cast<std::size_t>(indices[i]);
        auto idx1 = static_cast<std::size_t>(indices[i + 1]);
        auto idx2 = static_cast<std::size_t>(indices[i + 2]);

        const pxr::GfVec3f& p0 = positions[idx0];
        const pxr::GfVec3f& p1 = positions[idx1];
        const pxr::GfVec3f& p2 = positions[idx2];
        pxr::GfVec3f n = pxr::GfCross(p1 - p0, p2 - p0);
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

std::string makeAssetPath(const std::string& texturePath) {
    return fmt::format("{}/mem.cesium[{}]", GltfToUSD::CesiumMemLocation.generic_string(), texturePath);
}

pxr::SdfAssetPath convertTextureToUSD(
    const pxr::SdfPath& parentPath,
    const CesiumGltf::Model& model,
    const CesiumGltf::Texture& texture,
    int32_t textureIdx) {
    std::string texturePath = fmt::format("{}/texture_{}.bmp", parentPath.GetString(), textureIdx);

    auto textureSource = static_cast<std::size_t>(texture.source);
    const CesiumGltf::Image& img = model.images[textureSource];
    auto inMemoryAsset = std::make_shared<pxr::InMemoryAsset>(GltfToUSD::writeImageToBmp(img.cesium));
    auto& ctx = pxr::InMemoryAssetContext::instance();
    ctx.assets.insert({texturePath, std::move(inMemoryAsset)});

    return pxr::SdfAssetPath(makeAssetPath(texturePath));
}

pxr::UsdShadeMaterial convertMaterialToUSD(
    pxr::UsdStageRefPtr& stage,
    const pxr::SdfPath& parentPath,
    const std::vector<pxr::SdfAssetPath>& usdTexturePaths,
    const bool hasRasterOverlay,
    const pxr::SdfAssetPath& rasterOverlayPath,
    const CesiumGltf::Material& material,
    int32_t materialIdx) {
    std::string materialName = fmt::format("material_{}", materialIdx);
    pxr::SdfPath materialPath = parentPath.AppendChild(pxr::TfToken(materialName));
    pxr::UsdShadeMaterial materialUsd = pxr::UsdShadeMaterial::Define(stage, materialPath);

    const auto& pbrMetallicRoughness = material.pbrMetallicRoughness;
    pxr::UsdShadeShader pbrShader =
        pxr::UsdShadeShader::Define(stage, materialPath.AppendChild(pxr::TfToken("PBRShader")));
    pbrShader.CreateIdAttr(pxr::VtValue(pxr::_tokens->UsdShaderId));
    pbrShader.CreateInput(pxr::_tokens->roughness, pxr::SdfValueTypeNames->Float)
        .Set(float(pbrMetallicRoughness->roughnessFactor));
    pbrShader.CreateInput(pxr::_tokens->metallic, pxr::SdfValueTypeNames->Float)
        .Set(float(pbrMetallicRoughness->metallicFactor));

    std::vector<pxr::UsdShadeShader> stReaders(2);
    std::vector<pxr::UsdShadeInput> stInputs(2);
    for (std::size_t i = 0; i < 2; ++i) {
        stReaders[i] =
            pxr::UsdShadeShader::Define(stage, materialPath.AppendChild(pxr::TfToken(fmt::format("stReader_{}", i))));
        stReaders[i].CreateIdAttr(pxr::VtValue(pxr::_tokens->PrimStShaderId));

        stInputs[i] = materialUsd.CreateInput(pxr::_tokens->stPrimvarName, pxr::SdfValueTypeNames->Token);
        stInputs[i].Set(pxr::VtValue(pxr::TfToken(fmt::format("st_{}", i))));

        stReaders[i].CreateInput(pxr::_tokens->varname, pxr::SdfValueTypeNames->Token).ConnectToSource(stInputs[i]);
    }

    const auto setupDiffuseTexture =
        [&stage, &materialPath, &pbrShader, &stReaders](const pxr::SdfAssetPath& texturePath, const size_t texcoord) {
            const pxr::UsdShadeShader& stReader = stReaders[texcoord];
            pxr::UsdShadeShader diffuseTextureSampler =
                pxr::UsdShadeShader::Define(stage, materialPath.AppendChild(pxr::TfToken("DiffuseTexture")));
            diffuseTextureSampler.CreateIdAttr(pxr::VtValue(pxr::_tokens->UsdUVTexture));
            diffuseTextureSampler.CreateInput(pxr::_tokens->file, pxr::SdfValueTypeNames->Asset).Set(texturePath);
            diffuseTextureSampler.CreateInput(pxr::_tokens->st, pxr::SdfValueTypeNames->Float2)
                .ConnectToSource(stReader.ConnectableAPI(), pxr::_tokens->result);
            diffuseTextureSampler.CreateInput(pxr::_tokens->wrapS, pxr::SdfValueTypeNames->Token).Set(pxr::_tokens->clamp);
            diffuseTextureSampler.CreateInput(pxr::_tokens->wrapT, pxr::SdfValueTypeNames->Token).Set(pxr::_tokens->clamp);
            diffuseTextureSampler.CreateOutput(pxr::_tokens->rgb, pxr::SdfValueTypeNames->Float3);
            pbrShader.CreateInput(pxr::_tokens->diffuseColor, pxr::SdfValueTypeNames->Vector3f)
                .ConnectToSource(diffuseTextureSampler.ConnectableAPI(), pxr::_tokens->rgb);
        };

    if (hasRasterOverlay) {
        setupDiffuseTexture(rasterOverlayPath, 0);
    } else if (pbrMetallicRoughness->baseColorTexture) {
        auto baseColorIndex = static_cast<std::size_t>(pbrMetallicRoughness->baseColorTexture->index);
        const pxr::SdfAssetPath& texturePath = usdTexturePaths[baseColorIndex];
        auto baseColorTexCoord = static_cast<std::size_t>(pbrMetallicRoughness->baseColorTexture->texCoord);
        setupDiffuseTexture(texturePath, baseColorTexCoord);
    } else {
        pbrShader.CreateInput(pxr::_tokens->diffuseColor, pxr::SdfValueTypeNames->Vector3f)
            .Set(pxr::GfVec3f(
                static_cast<float>(pbrMetallicRoughness->baseColorFactor[0]),
                static_cast<float>(pbrMetallicRoughness->baseColorFactor[1]),
                static_cast<float>(pbrMetallicRoughness->baseColorFactor[2])));
    }

    materialUsd.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), pxr::_tokens->surface);

    return materialUsd;
}

void convertMeshToUSD(
    pxr::UsdStageRefPtr& stage,
    const pxr::SdfPath& parentPath,
    const CesiumGltf::Model& model,
    const CesiumGltf::Mesh& mesh,
    const std::vector<pxr::UsdShadeMaterial>& materials) {
    for (std::size_t i = 0; i < mesh.primitives.size(); ++i) {
        const CesiumGltf::MeshPrimitive& primitive = mesh.primitives[i];
        pxr::VtArray<pxr::GfVec3f> positions = getPrimitivePositions(model, primitive);
        pxr::VtArray<int> indices = getPrimitiveIndices(model, primitive, positions);
        pxr::VtArray<pxr::GfVec3f> normals = getPrimitiveNormals(model, primitive, positions, indices);
        pxr::VtArray<pxr::GfVec2f> st0 = getPrimitiveUVs(model, primitive, 0);
        pxr::VtArray<pxr::GfVec2f> st1 = getPrimitiveUVs(model, primitive, 1);
        pxr::VtArray<pxr::GfVec2f> rasterOverlaySt0 = getRasterOverlayUVs(model, primitive, 0);
        pxr::VtArray<int> faceVertexCounts(indices.size() / 3, 3);

        if (positions.empty() || indices.empty() || normals.empty()) {
            continue;
        }

        std::string meshName = fmt::format("mesh_primitive_{}", i);
        pxr::UsdGeomMesh meshUsd = pxr::UsdGeomMesh::Define(stage, parentPath.AppendChild(pxr::TfToken(meshName)));
        if (meshUsd) {
            meshUsd.CreateSubdivisionSchemeAttr().Set(pxr::UsdGeomTokens->none);
            meshUsd.CreateFaceVertexCountsAttr(pxr::VtValue::Take(faceVertexCounts));
            meshUsd.CreatePointsAttr(pxr::VtValue::Take(positions));
            meshUsd.CreateNormalsAttr(pxr::VtValue::Take(normals));
            meshUsd.SetNormalsInterpolation(pxr::UsdGeomTokens->constant);
            meshUsd.CreateFaceVertexIndicesAttr(pxr::VtValue::Take(indices));
            meshUsd.CreateDoubleSidedAttr().Set(true);
            if (!rasterOverlaySt0.empty()) {
                auto primVar = meshUsd.CreatePrimvar(pxr::_tokens->st_0, pxr::SdfValueTypeNames->TexCoord2fArray);
                primVar.SetInterpolation(pxr::_tokens->vertex);
                primVar.Set(rasterOverlaySt0);
            } else if (!st0.empty()) {
                auto primVar = meshUsd.CreatePrimvar(pxr::_tokens->st_0, pxr::SdfValueTypeNames->TexCoord2fArray);
                primVar.SetInterpolation(pxr::_tokens->vertex);
                primVar.Set(st0);
            }

            if (!st1.empty()) {
                auto primVar = meshUsd.CreatePrimvar(pxr::_tokens->st_1, pxr::SdfValueTypeNames->TexCoord2fArray);
                primVar.SetInterpolation(pxr::_tokens->vertex);
                primVar.Set(st1);
            }

            if (primitive.material >= 0) {
                pxr::UsdShadeMaterialBindingAPI usdMaterialBinding(meshUsd);

                auto primitiveMaterial = static_cast<std::size_t>(primitive.material);
                usdMaterialBinding.Bind(materials[primitiveMaterial]);
            }
        }
    }
}

void convertNodeToUSD(
    pxr::UsdStageRefPtr& stage,
    const pxr::SdfPath& parentPath,
    const CesiumGltf::Model& model,
    const CesiumGltf::Node& node,
    int32_t nodeIdx,
    const std::vector<pxr::UsdShadeMaterial>& materials) {
    std::string nodeName = fmt::format("node_{}", nodeIdx);
    pxr::SdfPath nodePath = parentPath.AppendChild(pxr::TfToken(nodeName));
    auto nodesSize = static_cast<int32_t>(model.nodes.size());
    for (std::int32_t child : node.children) {
        if (child >= 0 && child < nodesSize) {
            convertNodeToUSD(stage, nodePath, model, model.nodes[static_cast<std::size_t>(child)], child, materials);
        }
    }

    pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(stage, nodePath);
    if (!xform) {
        return;
    }

    pxr::GfMatrix4d currentTransform{1.0};
    if (node.matrix.size() == 16 && !isIdentityMatrix(node.matrix)) {
        currentTransform = pxr::GfMatrix4d(
            node.matrix[0],
            node.matrix[1],
            node.matrix[2],
            node.matrix[3],
            node.matrix[4],
            node.matrix[5],
            node.matrix[6],
            node.matrix[7],
            node.matrix[8],
            node.matrix[9],
            node.matrix[10],
            node.matrix[11],
            node.matrix[12],
            node.matrix[13],
            node.matrix[14],
            node.matrix[15]);
    } else {
        if (node.scale.size() == 3) {
            currentTransform.SetScale(pxr::GfVec3d(node.scale[0], node.scale[1], node.scale[2]));
        }

        if (node.rotation.size() == 4) {
            currentTransform.SetRotateOnly(
                pxr::GfQuatd(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]));
        }

        if (node.translation.size() == 3) {
            currentTransform.SetTranslateOnly(
                pxr::GfVec3d(node.translation[0], node.translation[1], node.translation[2]));
        }
    }

    xform.AddTransformOp().Set(currentTransform);

    auto meshIndex = static_cast<std::size_t>(node.mesh);
    if (meshIndex < model.meshes.size()) {
        convertMeshToUSD(stage, nodePath, model, model.meshes[meshIndex], materials);
    }
}

std::string getRasterOverlayTexturePath(const pxr::SdfPath& parentPath) {
    return fmt::format("{}/raster.bmp", parentPath.GetString());
}

} // namespace

std::filesystem::path GltfToUSD::CesiumMemLocation{};

std::vector<std::byte> GltfToUSD::writeImageToBmp(const CesiumGltf::ImageCesium& img) {
    std::vector<std::byte> writeData;
    stbi_write_bmp_to_func(
        [](void* context, void* data, int size) {
            auto& write = *reinterpret_cast<std::vector<std::byte>*>(context);
            std::byte* bdata = reinterpret_cast<std::byte*>(data);
            write.insert(write.end(), bdata, bdata + size);
        },
        &writeData,
        img.width,
        img.height,
        img.channels,
        img.pixelData.data());

    return writeData;
}

void GltfToUSD::insertRasterOverlayTexture(const pxr::SdfPath& parentPath, std::vector<std::byte>&& image) {
    std::string texturePath = getRasterOverlayTexturePath(parentPath);
    auto inMemoryAsset = std::make_shared<pxr::InMemoryAsset>(std::move(image));
    auto& ctx = pxr::InMemoryAssetContext::instance();
    ctx.assets.insert({texturePath, std::move(inMemoryAsset)});
}

pxr::UsdPrim GltfToUSD::convertToUSD(
    pxr::UsdStageRefPtr& stage,
    const pxr::SdfPath& modelPath,
    const CesiumGltf::Model& model,
    const glm::dmat4& matrix) {
    spdlog::default_logger()->info("convert to USD: {}", modelPath.GetString());

    std::vector<pxr::SdfAssetPath> textureUSDPaths;
    textureUSDPaths.reserve(model.textures.size());
    for (std::size_t i = 0; i < model.textures.size(); ++i) {
        textureUSDPaths.emplace_back(convertTextureToUSD(modelPath, model, model.textures[i], int32_t(i)));
    }

    const auto hasRasterOverlay = hasRasterOverlayUVs(model, 0);
    const auto rasterOverlayTexturePath = pxr::SdfAssetPath(makeAssetPath(getRasterOverlayTexturePath(modelPath)));

    std::vector<pxr::UsdShadeMaterial> materialUSDs;
    materialUSDs.reserve(model.materials.size());
    for (std::size_t i = 0; i < model.materials.size(); ++i) {
        materialUSDs.emplace_back(convertMaterialToUSD(
            stage,
            modelPath,
            textureUSDPaths,
            hasRasterOverlay,
            rasterOverlayTexturePath,
            model.materials[i],
            int32_t(i)));
    }

    pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(stage, modelPath);
    if (!xform) {
        return pxr::UsdPrim{};
    }

    pxr::GfMatrix4d currentTransform = pxr::GfMatrix4d(
        matrix[0].x,
        matrix[0].y,
        matrix[0].z,
        matrix[0].w,
        matrix[1].x,
        matrix[1].y,
        matrix[1].z,
        matrix[1].w,
        matrix[2].x,
        matrix[2].y,
        matrix[2].z,
        matrix[2].w,
        matrix[3].x,
        matrix[3].y,
        matrix[3].z,
        matrix[3].w);
    xform.AddTransformOp().Set(currentTransform);

    int32_t sceneIdx = model.scene;
    if (sceneIdx >= 0 && sceneIdx < static_cast<int32_t>(model.scenes.size())) {
        const CesiumGltf::Scene& scene = model.scenes[std::size_t(sceneIdx)];
        for (int32_t node : scene.nodes) {
            if (node >= 0 && model.nodes.size()) {
                convertNodeToUSD(stage, modelPath, model, model.nodes[std::size_t(node)], node, materialUSDs);
            }
        }
    } else if (!model.nodes.empty()) {
        convertNodeToUSD(stage, modelPath, model, model.nodes[0], 0, materialUSDs);
    } else {
        for (const auto& mesh : model.meshes) {
            convertMeshToUSD(stage, modelPath, model, mesh, materialUSDs);
        }
    }

    return xform.GetPrim();
}
} // namespace Cesium
