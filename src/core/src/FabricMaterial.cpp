#include "cesium/omniverse/FabricMaterial.h"

#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"
#include "cesium/omniverse/FabricTexture.h"
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

FabricMaterial::FabricMaterial(
    pxr::SdfPath path,
    const FabricMaterialDefinition& materialDefinition,
    pxr::SdfAssetPath defaultTextureAssetPath)
    : _materialDefinition(materialDefinition)
    , _defaultTextureAssetPath(std::move(defaultTextureAssetPath)) {

    initialize(std::move(path), materialDefinition);
    reset();
}

FabricMaterial::~FabricMaterial() {
    FabricUtil::destroyPrim(_materialPathFabric);
    FabricUtil::destroyPrim(_shaderPathFabric);

    if (_materialDefinition.hasBaseColorTexture()) {
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

    const auto materialPath = std::move(path);
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
            // clang-format off
            const auto vertexColorPrimvarNameSize = UsdTokens::vertexColor.GetString().size();
            srw.setArrayAttributeSize(shaderPathFabric, FabricTokens::inputs_vertex_color_name, vertexColorPrimvarNameSize);
            auto vertexColorNameFabric = srw.getArrayAttributeWr<uint8_t>(shaderPathFabric, FabricTokens::inputs_vertex_color_name);
            memcpy(vertexColorNameFabric.data(), UsdTokens::vertexColor.GetText(), vertexColorPrimvarNameSize);
            // clang-format on
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
            auto infoImplementationSourceFabric = srw.getAttributeWr<omni::fabric::Token>(baseColorTexPathFabric, FabricTokens::info_implementationSource);
            auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(baseColorTexPathFabric, FabricTokens::info_mdl_sourceAsset);
            auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(baseColorTexPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            auto paramColorSpaceFabric = srw.getArrayAttributeWr<omni::fabric::Token>(baseColorTexPathFabric, FabricTokens::_paramColorSpace);
            // clang-format on

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
}

void FabricMaterial::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();

    setMaterialValues(GltfUtil::getDefaultMaterialInfo());
    setTilesetIdAndTileId(-1, -1);

    if (_materialDefinition.hasBaseColorTexture()) {
        clearBaseColorTexture();
    }
}

void FabricMaterial::setMaterial(int64_t tilesetId, int64_t tileId, const MaterialInfo& materialInfo) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    setMaterialValues(materialInfo);
    setTilesetIdAndTileId(tilesetId, tileId);
}

void FabricMaterial::setBaseColorTexture(
    const std::shared_ptr<FabricTexture>& texture,
    const TextureInfo& textureInfo) {

    if (!UsdUtil::hasStage()) {
        return;
    }

    if (!_materialDefinition.hasBaseColorTexture()) {
        return;
    }

    setBaseColorTextureValues(texture->getAssetPath(), textureInfo);
}

void FabricMaterial::clearBaseColorTexture() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    setBaseColorTextureValues(_defaultTextureAssetPath, GltfUtil::getDefaultTextureInfo());
}

void FabricMaterial::setTilesetIdAndTileId(int64_t tilesetId, int64_t tileId) {
    FabricUtil::setTilesetIdAndTileId(_materialPathFabric, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_shaderPathFabric, tilesetId, tileId);

    if (_materialDefinition.hasBaseColorTexture()) {
        FabricUtil::setTilesetIdAndTileId(_baseColorTexPathFabric, tilesetId, tileId);
    }
}

void FabricMaterial::setMaterialValues(const MaterialInfo& materialInfo) {
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

    *alphaCutoffFabric = static_cast<float>(materialInfo.alphaCutoff);
    *alphaModeFabric = materialInfo.alphaMode;
    *baseAlphaFabric = static_cast<float>(materialInfo.baseAlpha);
    *baseColorFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.baseColorFactor));
    *emissiveFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.emissiveFactor));
    *metallicFactorFabric = static_cast<float>(materialInfo.metallicFactor);
    *roughnessFactorFabric = static_cast<float>(materialInfo.roughnessFactor);
}

void FabricMaterial::setBaseColorTextureValues(
    const pxr::SdfAssetPath& textureAssetPath,
    const TextureInfo& textureInfo) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    glm::dvec2 offset = textureInfo.offset;
    double rotation = textureInfo.rotation;
    glm::dvec2 scale = textureInfo.scale;

    if (!textureInfo.flipVertical) {
        // gltf/pbr.mdl does texture transform math in glTF coordinates (top-left origin), so we needed to convert
        // the translation and scale parameters to work in that space. This doesn't handle rotation yet because we
        // haven't needed it for imagery layers.
        offset = {offset.x, 1.0 - offset.y - scale.y};
        scale = {scale.x, scale.y};
    }

    // clang-format off
    auto textureFabric = srw.getAttributeWr<omni::fabric::AssetPath>(_baseColorTexPathFabric, FabricTokens::inputs_texture);
    auto texCoordIndexFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_tex_coord_index);
    auto wrapSFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_s);
    auto wrapTFabric = srw.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::inputs_wrap_t);
    auto offsetFabric = srw.getAttributeWr<pxr::GfVec2f>(_baseColorTexPathFabric, FabricTokens::inputs_offset);
    auto rotationFabric = srw.getAttributeWr<float>(_baseColorTexPathFabric, FabricTokens::inputs_rotation);
    auto scaleFabric = srw.getAttributeWr<pxr::GfVec2f>(_baseColorTexPathFabric, FabricTokens::inputs_scale);
    // clang-format on

    textureFabric->assetPath = pxr::TfToken(textureAssetPath.GetAssetPath());
    textureFabric->resolvedPath = pxr::TfToken(textureAssetPath.GetResolvedPath());
    *texCoordIndexFabric = static_cast<int>(textureInfo.setIndex);
    *wrapSFabric = textureInfo.wrapS;
    *wrapTFabric = textureInfo.wrapT;
    *offsetFabric = UsdUtil::glmToUsdVector(glm::fvec2(offset));
    *rotationFabric = static_cast<float>(rotation);
    *scaleFabric = UsdUtil::glmToUsdVector(glm::fvec2(scale));
}

} // namespace cesium::omniverse
