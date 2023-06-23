#include "cesium/omniverse/FabricMaterial.h"

#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/IFabric.h>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <spdlog/fmt/fmt.h>

#include <array>

namespace cesium::omniverse {

FabricMaterial::FabricMaterial(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition)
    : _materialDefinition(materialDefinition) {

    initialize(path, materialDefinition);

    // Remove this function once dynamic material values are supported in Kit 105
    setInitialValues(materialDefinition);
}

FabricMaterial::~FabricMaterial() {
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    FabricUtil::destroyPrim(_materialPathFabric);
    FabricUtil::destroyPrim(_shaderPathFabric);

    if (hasBaseColorTexture) {
        FabricUtil::destroyPrim(_baseColorTexPathFabric);
    }
}

void FabricMaterial::setActive(bool active) {
    if (!active) {
        reset();
    }
}

omni::fabric::Path FabricMaterial::getPathFabric() const {
    return _materialPathFabric;
}

const FabricMaterialDefinition& FabricMaterial::getMaterialDefinition() const {
    return _materialDefinition;
}

void FabricMaterial::initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition) {
    const auto hasBaseColorTexture = materialDefinition.hasBaseColorTexture();
    const auto hasVertexColors = materialDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto materialPath = path;
    const auto shaderPath = materialPath.AppendChild(UsdTokens::Shader);
    const auto baseColorTexPath = materialPath.AppendChild(UsdTokens::baseColorTex);

    const auto materialPathFabric = omni::fabric::Path(materialPath.GetText());
    const auto shaderPathFabric = omni::fabric::Path(shaderPath.GetText());
    const auto baseColorTexPathFabric = omni::fabric::Path(baseColorTexPath.GetText());

    // Material
    {
        srw.createPrim(materialPathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(materialPathFabric);
    }

    // Shader
    {
        srw.createPrim(shaderPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::inputs_alpha_cutoff, FabricTokens::inputs_alpha_cutoff);
        attributes.addAttribute(FabricTypes::inputs_alpha_mode, FabricTokens::inputs_alpha_mode);
        attributes.addAttribute(FabricTypes::inputs_base_alpha, FabricTokens::inputs_base_alpha);
        attributes.addAttribute(FabricTypes::inputs_base_color_factor, FabricTokens::inputs_base_color_factor);
        attributes.addAttribute(FabricTypes::inputs_emissive_factor, FabricTokens::inputs_emissive_factor);
        attributes.addAttribute(FabricTypes::inputs_metallic_factor, FabricTokens::inputs_metallic_factor);
        attributes.addAttribute(FabricTypes::inputs_roughness_factor, FabricTokens::inputs_roughness_factor);
        attributes.addAttribute(FabricTypes::outputs_out, FabricTokens::outputs_out);
        attributes.addAttribute(FabricTypes::info_implementationSource, FabricTokens::info_implementationSource);
        attributes.addAttribute(FabricTypes::info_mdl_sourceAsset, FabricTokens::info_mdl_sourceAsset);
        attributes.addAttribute(FabricTypes::info_mdl_sourceAsset_subIdentifier, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_sdrMetadata, FabricTokens::_sdrMetadata);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        if (hasVertexColors) {
            attributes.addAttribute(FabricTypes::inputs_vertex_color_name, FabricTokens::inputs_vertex_color_name);
        }

        attributes.createAttributes(shaderPathFabric);

        srw.setArrayAttributeSize(shaderPathFabric, FabricTokens::_paramColorSpace, 0);
        srw.setArrayAttributeSize(shaderPathFabric, FabricTokens::_sdrMetadata, 0);

        // clang-format off
        auto infoImplementationSourceFabric = srw.getAttributeWr<omni::fabric::Token>(shaderPathFabric, FabricTokens::info_implementationSource);
        auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(shaderPathFabric, FabricTokens::info_mdl_sourceAsset);
        auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(shaderPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        // clang-format on

        *infoImplementationSourceFabric = FabricTokens::sourceAsset;
        infoMdlSourceAssetFabric->assetPath = UsdTokens::gltf_pbr_mdl;
        infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
        *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::gltf_material;

        if (hasVertexColors) {
            const auto vertexColorPrimvarNameSize = UsdTokens::vertexColor.GetString().size();
            srw.setArrayAttributeSize(
                shaderPathFabric, FabricTokens::inputs_vertex_color_name, vertexColorPrimvarNameSize);
            auto vertexColorNameFabric =
                srw.getArrayAttributeWr<uint8_t>(shaderPathFabric, FabricTokens::inputs_vertex_color_name);
            memcpy(vertexColorNameFabric.data(), UsdTokens::vertexColor.GetText(), vertexColorPrimvarNameSize);
        }

        // Connect the material terminals to the shader.
        srw.createConnection(
            materialPathFabric,
            FabricTokens::outputs_mdl_surface,
            omni::fabric::Connection{omni::fabric::PathC(shaderPathFabric), FabricTokens::outputs_out});
        srw.createConnection(
            materialPathFabric,
            FabricTokens::outputs_mdl_displacement,
            omni::fabric::Connection{omni::fabric::PathC(shaderPathFabric), FabricTokens::outputs_out});
        srw.createConnection(
            materialPathFabric,
            FabricTokens::outputs_mdl_volume,
            omni::fabric::Connection{omni::fabric::PathC(shaderPathFabric), FabricTokens::outputs_out});
    }

    if (hasBaseColorTexture) {
        // Create the base color texture
        const auto baseColorTextureName =
            fmt::format("{}_inputs_base_color_texture", UsdUtil::getSafeName(materialPath.GetString()));
        const auto baseColorTexturePath =
            pxr::SdfAssetPath(fmt::format("{}{}", rtx::resourcemanager::kDynamicTexturePrefix, baseColorTextureName));
        _baseColorTexture = std::make_unique<omni::ui::DynamicTextureProvider>(baseColorTextureName);

        // baseColorTex
        {
            srw.createPrim(baseColorTexPathFabric);

            FabricAttributesBuilder attributes;

            // clang-format off
            attributes.addAttribute(FabricTypes::inputs_offset, FabricTokens::inputs_offset);
            attributes.addAttribute(FabricTypes::inputs_rotation, FabricTokens::inputs_rotation);
            attributes.addAttribute(FabricTypes::inputs_scale, FabricTokens::inputs_scale);
            attributes.addAttribute(FabricTypes::inputs_tex_coord_index, FabricTokens::inputs_tex_coord_index);
            attributes.addAttribute(FabricTypes::inputs_texture, FabricTokens::inputs_texture);
            attributes.addAttribute(FabricTypes::inputs_wrap_s, FabricTokens::inputs_wrap_s);
            attributes.addAttribute(FabricTypes::inputs_wrap_t, FabricTokens::inputs_wrap_t);
            attributes.addAttribute(FabricTypes::outputs_out, FabricTokens::outputs_out);
            attributes.addAttribute(FabricTypes::info_implementationSource, FabricTokens::info_implementationSource);
            attributes.addAttribute(FabricTypes::info_mdl_sourceAsset, FabricTokens::info_mdl_sourceAsset);
            attributes.addAttribute(FabricTypes::info_mdl_sourceAsset_subIdentifier, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
            attributes.addAttribute(FabricTypes::_sdrMetadata, FabricTokens::_sdrMetadata);
            attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
            attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
            attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
            // clang-format on

            attributes.createAttributes(baseColorTexPathFabric);

            // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
            srw.setArrayAttributeSize(baseColorTexPathFabric, FabricTokens::_paramColorSpace, 2);
            srw.setArrayAttributeSize(baseColorTexPathFabric, FabricTokens::_sdrMetadata, 0);

            // clang-format off
            auto offsetFabric = srw.getAttributeWr<pxr::GfVec2f>(baseColorTexPathFabric, FabricTokens::inputs_offset);
            auto rotationFabric = srw.getAttributeWr<float>(baseColorTexPathFabric, FabricTokens::inputs_rotation);
            auto scaleFabric = srw.getAttributeWr<pxr::GfVec2f>(baseColorTexPathFabric, FabricTokens::inputs_scale);
            auto texCoordIndexFabric = srw.getAttributeWr<int>(baseColorTexPathFabric, FabricTokens::inputs_tex_coord_index);
            auto textureFabric = srw.getAttributeWr<omni::fabric::AssetPath>(baseColorTexPathFabric, FabricTokens::inputs_texture);
            auto infoImplementationSourceFabric = srw.getAttributeWr<omni::fabric::Token>(baseColorTexPathFabric, FabricTokens::info_implementationSource);
            auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(baseColorTexPathFabric, FabricTokens::info_mdl_sourceAsset);
            auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(baseColorTexPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            auto paramColorSpaceFabric = srw.getArrayAttributeWr<omni::fabric::Token>(baseColorTexPathFabric, FabricTokens::_paramColorSpace);
            // clang-format on

            *offsetFabric = pxr::GfVec2f(0.0f, 0.0f);
            *rotationFabric = 0.0f;
            *scaleFabric = pxr::GfVec2f(1.0f, 1.0f);
            *texCoordIndexFabric = 0;
            textureFabric->assetPath = pxr::TfToken(baseColorTexturePath.GetAssetPath());
            textureFabric->resolvedPath = pxr::TfToken(baseColorTexturePath.GetResolvedPath());
            *infoImplementationSourceFabric = FabricTokens::sourceAsset;
            infoMdlSourceAssetFabric->assetPath = UsdTokens::gltf_pbr_mdl;
            infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
            *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::gltf_texture_lookup;
            paramColorSpaceFabric[0] = FabricTokens::inputs_texture;
            paramColorSpaceFabric[1] = FabricTokens::_auto;

            // Create connection from shader to texture.
            srw.createConnection(
                shaderPathFabric,
                FabricTokens::inputs_base_color_texture,
                omni::fabric::Connection{omni::fabric::PathC(baseColorTexPathFabric), FabricTokens::outputs_out});
        }
    }

    _materialPathFabric = materialPathFabric;
    _shaderPathFabric = shaderPathFabric;
    _baseColorTexPathFabric = baseColorTexPathFabric;

    reset();
}

void FabricMaterial::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    if (hasBaseColorTexture) {
        // Clear the texture
        const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
        const auto size = carb::Uint2{1, 1};
        _baseColorTexture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
    }

    const auto alphaCutoff = GltfUtil::getDefaultAlphaCutoff();
    const auto alphaMode = GltfUtil::getDefaultAlphaMode();
    const auto baseAlpha = GltfUtil::getDefaultBaseAlpha();
    const auto baseColorFactor = GltfUtil::getDefaultBaseColorFactor();
    const auto emissiveFactor = GltfUtil::getDefaultEmissiveFactor();
    const auto metallicFactor = GltfUtil::getDefaultMetallicFactor();
    const auto roughnessFactor = GltfUtil::getDefaultRoughnessFactor();
    const auto baseColorTextureWrapS = GltfUtil::getDefaultWrapS();
    const auto baseColorTextureWrapT = GltfUtil::getDefaultWrapT();

    // clang-format off
    auto alphaCutoffFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_alpha_cutoff);
    auto alphaModeFabric = srw.getAttributeWr<int>(_shaderPathFabric, FabricTokens::inputs_alpha_mode);
    auto baseAlphaFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_base_alpha);
    auto baseColorFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::inputs_base_color_factor);
    auto emissiveFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::inputs_emissive_factor);
    auto metallicFactorFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_metallic_factor);
    auto roughnessFactorFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_roughness_factor);
    // clang-format on

    *alphaCutoffFabric = alphaCutoff;
    *alphaModeFabric = alphaMode;
    *baseAlphaFabric = baseAlpha;
    *baseColorFactorFabric = baseColorFactor;
    *emissiveFactorFabric = emissiveFactor;
    *metallicFactorFabric = metallicFactor;
    *roughnessFactorFabric = roughnessFactor;

    if (hasBaseColorTexture) {
        auto wrapSFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_s);
        auto wrapTFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_t);

        *wrapSFabric = baseColorTextureWrapS;
        *wrapTFabric = baseColorTextureWrapT;
    }

    FabricUtil::setTilesetIdAndTileId(_materialPathFabric, -1, -1);
    FabricUtil::setTilesetIdAndTileId(_shaderPathFabric, -1, -1);

    if (hasBaseColorTexture) {
        FabricUtil::setTilesetIdAndTileId(_baseColorTexPathFabric, -1, -1);
    }
}

void FabricMaterial::setInitialValues(const FabricMaterialDefinition& materialDefinition) {
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    // clang-format off
    auto alphaCutoffFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_alpha_cutoff);
    auto alphaModeFabric = srw.getAttributeWr<int>(_shaderPathFabric, FabricTokens::inputs_alpha_mode);
    auto baseAlphaFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_base_alpha);
    auto baseColorFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::inputs_base_color_factor);
    auto emissiveFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::inputs_emissive_factor);
    auto metallicFactorFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_metallic_factor);
    auto roughnessFactorFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_roughness_factor);
    // clang-format on

    *alphaCutoffFabric = materialDefinition.getAlphaCutoff();
    *alphaModeFabric = materialDefinition.getAlphaMode();
    *baseAlphaFabric = materialDefinition.getBaseAlpha();
    *baseColorFactorFabric = materialDefinition.getBaseColorFactor();
    *emissiveFactorFabric = materialDefinition.getEmissiveFactor();
    *metallicFactorFabric = materialDefinition.getMetallicFactor();
    *roughnessFactorFabric = materialDefinition.getRoughnessFactor();

    if (hasBaseColorTexture) {
        auto wrapSFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_s);
        auto wrapTFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_t);

        *wrapSFabric = materialDefinition.getWrapS();
        *wrapTFabric = materialDefinition.getWrapT();
    }
}

void FabricMaterial::setTile(
    int64_t tilesetId,
    int64_t tileId,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::ImageCesium* imagery) {

    auto srw = UsdUtil::getFabricStageReaderWriter();

    float alphaCutoff;
    int alphaMode;
    float baseAlpha;
    pxr::GfVec3f baseColorFactor;
    pxr::GfVec3f emissiveFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureWrapS;
    int baseColorTextureWrapT;

    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();
    const auto hasGltfMaterial = GltfUtil::hasMaterial(primitive);

    const CesiumGltf::ImageCesium* baseColorImage = nullptr;

    if (hasGltfMaterial) {
        const auto& material = model.materials[static_cast<size_t>(primitive.material)];

        alphaCutoff = GltfUtil::getAlphaCutoff(material);
        alphaMode = GltfUtil::getAlphaMode(material);
        baseAlpha = GltfUtil::getBaseAlpha(material);
        baseColorFactor = GltfUtil::getBaseColorFactor(material);
        emissiveFactor = GltfUtil::getBaseColorFactor(material);
        metallicFactor = GltfUtil::getMetallicFactor(material);
        roughnessFactor = GltfUtil::getRoughnessFactor(material);
        baseColorTextureWrapS = GltfUtil::getBaseColorTextureWrapS(model, material);
        baseColorTextureWrapT = GltfUtil::getBaseColorTextureWrapT(model, material);

        const auto baseColorTextureIndex = GltfUtil::getBaseColorTextureIndex(model, material);

        if (baseColorTextureIndex.has_value()) {
            baseColorImage = &GltfUtil::getImageCesium(model, model.textures[baseColorTextureIndex.value()]);
        }
    } else {
        alphaCutoff = GltfUtil::getDefaultAlphaCutoff();
        alphaMode = GltfUtil::getDefaultAlphaMode();
        baseAlpha = GltfUtil::getDefaultBaseAlpha();
        baseColorFactor = GltfUtil::getDefaultBaseColorFactor();
        emissiveFactor = GltfUtil::getDefaultEmissiveFactor();
        metallicFactor = GltfUtil::getDefaultMetallicFactor();
        roughnessFactor = GltfUtil::getDefaultRoughnessFactor();
        baseColorTextureWrapS = GltfUtil::getDefaultWrapS();
        baseColorTextureWrapT = GltfUtil::getDefaultWrapT();
    }

    // Imagery overrides the base color texture in the glTF
    if (imagery != nullptr) {
        baseColorImage = imagery;
        baseColorTextureWrapS = CesiumGltf::Sampler::WrapS::CLAMP_TO_EDGE;
        baseColorTextureWrapT = CesiumGltf::Sampler::WrapS::CLAMP_TO_EDGE;
    }

    if (hasBaseColorTexture) {
        _baseColorTexture->setBytesData(
            reinterpret_cast<const uint8_t*>(baseColorImage->pixelData.data()),
            carb::Uint2{static_cast<uint32_t>(baseColorImage->width), static_cast<uint32_t>(baseColorImage->height)},
            omni::ui::kAutoCalculateStride,
            carb::Format::eRGBA8_SRGB);
    }

    // clang-format off
    auto alphaCutoffFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_alpha_cutoff);
    auto alphaModeFabric = srw.getAttributeWr<int>(_shaderPathFabric, FabricTokens::inputs_alpha_mode);
    auto baseAlphaFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_base_alpha);
    auto baseColorFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::inputs_base_color_factor);
    auto emissiveFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::inputs_emissive_factor);
    auto metallicFactorFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_metallic_factor);
    auto roughnessFactorFabric = srw.getAttributeWr<float>(_shaderPathFabric, FabricTokens::inputs_roughness_factor);
    // clang-format on

    *alphaCutoffFabric = alphaCutoff;
    *alphaModeFabric = alphaMode;
    *baseAlphaFabric = baseAlpha;
    *baseColorFactorFabric = baseColorFactor;
    *emissiveFactorFabric = emissiveFactor;
    *metallicFactorFabric = metallicFactor;
    *roughnessFactorFabric = roughnessFactor;

    if (hasBaseColorTexture) {
        auto wrapSFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_s);
        auto wrapTFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_t);

        *wrapSFabric = baseColorTextureWrapS;
        *wrapTFabric = baseColorTextureWrapT;
    }

    FabricUtil::setTilesetIdAndTileId(_materialPathFabric, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_shaderPathFabric, tilesetId, tileId);

    if (hasBaseColorTexture) {
        FabricUtil::setTilesetIdAndTileId(_baseColorTexPathFabric, tilesetId, tileId);
    }
}
}; // namespace cesium::omniverse
