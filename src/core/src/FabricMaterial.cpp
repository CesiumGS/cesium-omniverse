#include "cesium/omniverse/FabricMaterial.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/fabric/FabricUSD.h>
#include <spdlog/fmt/fmt.h>

namespace cesium::omniverse {

FabricMaterial::FabricMaterial(
    const pxr::SdfPath& path,
    const FabricMaterialDefinition& materialDefinition,
    const pxr::TfToken& defaultTextureAssetPathToken)
    : _materialPath(FabricUtil::toFabricPath(path))
    , _materialDefinition(materialDefinition)
    , _defaultTextureAssetPathToken(defaultTextureAssetPathToken) {

    if (materialDefinition.hasTilesetMaterial()) {
        const auto tilesetMaterialPath = FabricUtil::toFabricPath(materialDefinition.getTilesetMaterialPath());
        initializeFromExistingMaterial(tilesetMaterialPath);
    } else {
        initialize();
    }

    reset();
}

FabricMaterial::~FabricMaterial() {
    for (const auto& path : _allPaths) {
        FabricUtil::destroyPrim(path);
    }
}

void FabricMaterial::setActive(bool active) {
    if (!active) {
        reset();
    }
}

omni::fabric::Path FabricMaterial::getPathFabric() const {
    return _materialPath;
}

const FabricMaterialDefinition& FabricMaterial::getMaterialDefinition() const {
    return _materialDefinition;
}

void FabricMaterial::initialize() {
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    const auto& materialPath = _materialPath;
    const auto shaderPath = FabricUtil::joinPaths(materialPath, FabricTokens::Shader);
    const auto baseColorTexturePath = FabricUtil::joinPaths(materialPath, FabricTokens::baseColorTex);

    createMaterial(materialPath);
    _allPaths.push_back(materialPath);

    createShader(shaderPath, materialPath);
    _allPaths.push_back(shaderPath);
    _shaderPaths.push_back(shaderPath);

    if (hasBaseColorTexture) {
        createTexture(baseColorTexturePath, shaderPath, FabricTokens::inputs_base_color_texture);
        _allPaths.push_back(baseColorTexturePath);
        _baseColorTexturePaths.push_back(baseColorTexturePath);
    }
}

void FabricMaterial::initializeFromExistingMaterial(const omni::fabric::Path& srcMaterialPath) {
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto& dstMaterialPath = _materialPath;

    const auto dstPaths = FabricUtil::copyMaterial(srcMaterialPath, dstMaterialPath);

    for (const auto& dstPath : dstPaths) {
        srw.createAttribute(dstPath, FabricTokens::_cesium_tilesetId, FabricTypes::_cesium_tilesetId);
        _allPaths.push_back(dstPath);

        const auto mdlIdentifier = FabricUtil::getMdlIdentifier(dstPath);

        if (mdlIdentifier == FabricTokens::cesium_base_color_texture) {
            if (hasBaseColorTexture) {
                // Create a base color texture node to fill the empty slot
                const auto baseColorTexturePath = FabricUtil::joinPaths(dstMaterialPath, FabricTokens::baseColorTex);
                createTexture(baseColorTexturePath, dstPath, FabricTokens::inputs_base_color_texture);
                _allPaths.push_back(baseColorTexturePath);
                _baseColorTexturePaths.push_back(baseColorTexturePath);
            }
        }
    }
}

void FabricMaterial::createMaterial(const omni::fabric::Path& materialPath) {
    auto srw = UsdUtil::getFabricStageReaderWriter();
    srw.createPrim(materialPath);

    FabricAttributesBuilder attributes;

    attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);

    attributes.createAttributes(materialPath);
}

void FabricMaterial::createShader(const omni::fabric::Path& shaderPath, const omni::fabric::Path& materialPath) {
    const auto hasVertexColors = _materialDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(shaderPath);

    FabricAttributesBuilder attributes;

    // clang-format off
    attributes.addAttribute(FabricTypes::inputs_alpha_cutoff, FabricTokens::inputs_alpha_cutoff);
    attributes.addAttribute(FabricTypes::inputs_alpha_mode, FabricTokens::inputs_alpha_mode);
    attributes.addAttribute(FabricTypes::inputs_base_alpha, FabricTokens::inputs_base_alpha);
    attributes.addAttribute(FabricTypes::inputs_base_color_factor, FabricTokens::inputs_base_color_factor);
    attributes.addAttribute(FabricTypes::inputs_emissive_factor, FabricTokens::inputs_emissive_factor);
    attributes.addAttribute(FabricTypes::inputs_metallic_factor, FabricTokens::inputs_metallic_factor);
    attributes.addAttribute(FabricTypes::inputs_roughness_factor, FabricTokens::inputs_roughness_factor);
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

    if (hasVertexColors) {
        attributes.addAttribute(FabricTypes::inputs_vertex_color_name, FabricTokens::inputs_vertex_color_name);
    }

    attributes.createAttributes(shaderPath);

    srw.setArrayAttributeSize(shaderPath, FabricTokens::_paramColorSpace, 0);
    srw.setArrayAttributeSize(shaderPath, FabricTokens::_sdrMetadata, 0);

    // clang-format off
    auto inputsExcludeFromWhiteModeFabric = srw.getAttributeWr<bool>(shaderPath, FabricTokens::inputs_excludeFromWhiteMode);
    auto infoImplementationSourceFabric = srw.getAttributeWr<omni::fabric::Token>(shaderPath, FabricTokens::info_implementationSource);
    auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(shaderPath, FabricTokens::info_mdl_sourceAsset);
    auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(shaderPath, FabricTokens::info_mdl_sourceAsset_subIdentifier);
    // clang-format on

    *inputsExcludeFromWhiteModeFabric = false;
    *infoImplementationSourceFabric = FabricTokens::sourceAsset;
    infoMdlSourceAssetFabric->assetPath = UsdTokens::gltf_pbr_mdl;
    infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
    *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::gltf_material;

    if (hasVertexColors) {
        const auto vertexColorPrimvarNameSize = UsdTokens::vertexColor.GetString().size();
        srw.setArrayAttributeSize(shaderPath, FabricTokens::inputs_vertex_color_name, vertexColorPrimvarNameSize);
        auto vertexColorNameFabric =
            srw.getArrayAttributeWr<uint8_t>(shaderPath, FabricTokens::inputs_vertex_color_name);
        memcpy(vertexColorNameFabric.data(), UsdTokens::vertexColor.GetText(), vertexColorPrimvarNameSize);
    }

    // Connect the material terminals to the shader.
    srw.createConnection(
        materialPath,
        FabricTokens::outputs_mdl_surface,
        omni::fabric::Connection{omni::fabric::PathC(shaderPath), FabricTokens::outputs_out});
    srw.createConnection(
        materialPath,
        FabricTokens::outputs_mdl_displacement,
        omni::fabric::Connection{omni::fabric::PathC(shaderPath), FabricTokens::outputs_out});
    srw.createConnection(
        materialPath,
        FabricTokens::outputs_mdl_volume,
        omni::fabric::Connection{omni::fabric::PathC(shaderPath), FabricTokens::outputs_out});
}

void FabricMaterial::createTexture(
    const omni::fabric::Path& texturePath,
    const omni::fabric::Path& shaderPath,
    const omni::fabric::Token& shaderInput) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(texturePath);

    FabricAttributesBuilder attributes;

    // clang-format off
    attributes.addAttribute(FabricTypes::inputs_offset, FabricTokens::inputs_offset);
    attributes.addAttribute(FabricTypes::inputs_rotation, FabricTokens::inputs_rotation);
    attributes.addAttribute(FabricTypes::inputs_scale, FabricTokens::inputs_scale);
    attributes.addAttribute(FabricTypes::inputs_tex_coord_index, FabricTokens::inputs_tex_coord_index);
    attributes.addAttribute(FabricTypes::inputs_texture, FabricTokens::inputs_texture);
    attributes.addAttribute(FabricTypes::inputs_wrap_s, FabricTokens::inputs_wrap_s);
    attributes.addAttribute(FabricTypes::inputs_wrap_t, FabricTokens::inputs_wrap_t);
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

    attributes.createAttributes(texturePath);

    // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
    srw.setArrayAttributeSize(texturePath, FabricTokens::_paramColorSpace, 2);
    srw.setArrayAttributeSize(texturePath, FabricTokens::_sdrMetadata, 0);

    // clang-format off
    auto inputsExcludeFromWhiteModeFabric = srw.getAttributeWr<bool>(texturePath, FabricTokens::inputs_excludeFromWhiteMode);
    auto infoImplementationSourceFabric = srw.getAttributeWr<omni::fabric::Token>(texturePath, FabricTokens::info_implementationSource);
    auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(texturePath, FabricTokens::info_mdl_sourceAsset);
    auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(texturePath, FabricTokens::info_mdl_sourceAsset_subIdentifier);
    auto paramColorSpaceFabric = srw.getArrayAttributeWr<omni::fabric::Token>(texturePath, FabricTokens::_paramColorSpace);
    // clang-format on

    // For some reason, when a material network has both cesium nodes and glTF nodes it causes an infinite loop of
    // material loading. If this material is initialized from a tileset material use the cesium wrapper nodes instead.
    const auto& custom = _materialDefinition.hasTilesetMaterial();
    const auto& assetPath = custom ? Context::instance().getCesiumMdlPathToken() : UsdTokens::gltf_pbr_mdl;
    const auto& subIdentifier = custom ? FabricTokens::cesium_texture_lookup : FabricTokens::gltf_texture_lookup;

    *inputsExcludeFromWhiteModeFabric = false;
    *infoImplementationSourceFabric = FabricTokens::sourceAsset;
    infoMdlSourceAssetFabric->assetPath = assetPath;
    infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
    *infoMdlSourceAssetSubIdentifierFabric = subIdentifier;
    paramColorSpaceFabric[0] = FabricTokens::inputs_texture;
    paramColorSpaceFabric[1] = FabricTokens::_auto;

    // Create connection from shader to texture.
    srw.createConnection(
        shaderPath, shaderInput, omni::fabric::Connection{omni::fabric::PathC(texturePath), FabricTokens::outputs_out});
}

void FabricMaterial::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    clearMaterial();
    clearBaseColorTexture();
}

void FabricMaterial::setMaterial(int64_t tilesetId, const MaterialInfo& materialInfo) {
    for (auto& shaderPath : _shaderPaths) {
        setShaderValues(shaderPath, materialInfo);
    }

    for (const auto& path : _allPaths) {
        FabricUtil::setTilesetId(path, tilesetId);
    }
}

void FabricMaterial::setBaseColorTexture(const pxr::TfToken& textureAssetPathToken, const TextureInfo& textureInfo) {
    for (auto& _baseColorTexturePath : _baseColorTexturePaths) {
        setTextureValues(_baseColorTexturePath, textureAssetPathToken, textureInfo);
    }
}

void FabricMaterial::clearMaterial() {
    setMaterial(NO_TILESET_ID, GltfUtil::getDefaultMaterialInfo());
}

void FabricMaterial::clearBaseColorTexture() {
    setBaseColorTexture(_defaultTextureAssetPathToken, GltfUtil::getDefaultTextureInfo());
}

void FabricMaterial::setShaderValues(const omni::fabric::Path& shaderPath, const MaterialInfo& materialInfo) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    auto alphaCutoffFabric = srw.getAttributeWr<float>(shaderPath, FabricTokens::inputs_alpha_cutoff);
    auto alphaModeFabric = srw.getAttributeWr<int>(shaderPath, FabricTokens::inputs_alpha_mode);
    auto baseAlphaFabric = srw.getAttributeWr<float>(shaderPath, FabricTokens::inputs_base_alpha);
    auto baseColorFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(shaderPath, FabricTokens::inputs_base_color_factor);
    auto emissiveFactorFabric = srw.getAttributeWr<pxr::GfVec3f>(shaderPath, FabricTokens::inputs_emissive_factor);
    auto metallicFactorFabric = srw.getAttributeWr<float>(shaderPath, FabricTokens::inputs_metallic_factor);
    auto roughnessFactorFabric = srw.getAttributeWr<float>(shaderPath, FabricTokens::inputs_roughness_factor);

    *alphaCutoffFabric = static_cast<float>(materialInfo.alphaCutoff);
    *alphaModeFabric = materialInfo.alphaMode;
    *baseAlphaFabric = static_cast<float>(materialInfo.baseAlpha);
    *baseColorFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.baseColorFactor));
    *emissiveFactorFabric = UsdUtil::glmToUsdVector(glm::fvec3(materialInfo.emissiveFactor));
    *metallicFactorFabric = static_cast<float>(materialInfo.metallicFactor);
    *roughnessFactorFabric = static_cast<float>(materialInfo.roughnessFactor);
}

void FabricMaterial::setTextureValues(
    const omni::fabric::Path& texturePath,
    const pxr::TfToken& textureAssetPathToken,
    const TextureInfo& textureInfo) {
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

    auto textureFabric = srw.getAttributeWr<omni::fabric::AssetPath>(texturePath, FabricTokens::inputs_texture);
    auto texCoordIndexFabric = srw.getAttributeWr<int>(texturePath, FabricTokens::inputs_tex_coord_index);
    auto wrapSFabric = srw.getAttributeWr<int>(texturePath, FabricTokens::inputs_wrap_s);
    auto wrapTFabric = srw.getAttributeWr<int>(texturePath, FabricTokens::inputs_wrap_t);
    auto offsetFabric = srw.getAttributeWr<pxr::GfVec2f>(texturePath, FabricTokens::inputs_offset);
    auto rotationFabric = srw.getAttributeWr<float>(texturePath, FabricTokens::inputs_rotation);
    auto scaleFabric = srw.getAttributeWr<pxr::GfVec2f>(texturePath, FabricTokens::inputs_scale);

    textureFabric->assetPath = textureAssetPathToken;
    textureFabric->resolvedPath = pxr::TfToken();
    *texCoordIndexFabric = static_cast<int>(textureInfo.setIndex);
    *wrapSFabric = textureInfo.wrapS;
    *wrapTFabric = textureInfo.wrapT;
    *offsetFabric = UsdUtil::glmToUsdVector(glm::fvec2(offset));
    *rotationFabric = static_cast<float>(rotation);
    *scaleFabric = UsdUtil::glmToUsdVector(glm::fvec2(scale));
}

} // namespace cesium::omniverse
