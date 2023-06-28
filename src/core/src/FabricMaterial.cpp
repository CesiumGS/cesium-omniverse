#include "cesium/omniverse/FabricMaterial.h"

#include "cesium/omniverse/FabricAsset.h"
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
#include <carb/flatcache/FlatCacheUSD.h>
#include <omni/ui/ImageProvider/DynamicTextureProvider.h>
#include <spdlog/fmt/fmt.h>

#include <array>

namespace cesium::omniverse {

FabricMaterial::FabricMaterial(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition)
    : _materialDefinition(materialDefinition) {

    initialize(std::move(path), materialDefinition);

    // Remove this function once dynamic material values are supported in Kit 105
    setInitialValues(materialDefinition);
}

FabricMaterial::~FabricMaterial() {
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    FabricUtil::destroyPrim(_materialPathFabric);
    FabricUtil::destroyPrim(_shaderPathFabric);
    FabricUtil::destroyPrim(_displacementPathFabric);
    FabricUtil::destroyPrim(_surfacePathFabric);

    if (hasBaseColorTexture) {
        FabricUtil::destroyPrim(_baseColorTexPathFabric);
    }
}

void FabricMaterial::setActive(bool active) {
    if (!active) {
        reset();
    }
}

carb::flatcache::Path FabricMaterial::getPathFabric() const {
    return _materialPathFabric;
}

const FabricMaterialDefinition& FabricMaterial::getMaterialDefinition() const {
    return _materialDefinition;
}

void FabricMaterial::initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition) {
    const auto hasBaseColorTexture = materialDefinition.hasBaseColorTexture();
    const auto hasVertexColors = materialDefinition.hasVertexColors();

    auto sip = UsdUtil::getFabricStageInProgress();

    const auto materialPath = std::move(path);
    const auto shaderPath = materialPath.AppendChild(UsdTokens::Shader);
    const auto displacementPath = materialPath.AppendChild(UsdTokens::displacement);
    const auto surfacePath = materialPath.AppendChild(UsdTokens::surface);
    const auto baseColorTexPath = materialPath.AppendChild(UsdTokens::baseColorTex);

    const auto materialPathFabric = carb::flatcache::Path(materialPath.GetText());
    const auto displacementPathFabric = carb::flatcache::Path(displacementPath.GetText());
    const auto surfacePathFabric = carb::flatcache::Path(surfacePath.GetText());
    const auto shaderPathFabric = carb::flatcache::Path(shaderPath.GetText());
    const auto baseColorTexPathFabric = carb::flatcache::Path(baseColorTexPath.GetText());

    const auto shaderPathFabricUint64 = carb::flatcache::PathC(shaderPathFabric).path;
    const auto baseColorTexPathFabricUint64 = carb::flatcache::PathC(baseColorTexPathFabric).path;

    // Material
    {
        sip.createPrim(materialPathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::_terminals, FabricTokens::_terminals);
        attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(materialPathFabric);

        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminals, 2);

        auto terminalsFabric = sip.getArrayAttributeWr<uint64_t>(materialPathFabric, FabricTokens::_terminals);

        terminalsFabric[0] = shaderPathFabricUint64;
        terminalsFabric[1] = shaderPathFabricUint64;
    }

    // Displacement
    {
        sip.createPrim(displacementPathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::_nodePaths, FabricTokens::_nodePaths);
        attributes.addAttribute(FabricTypes::_relationships_inputId, FabricTokens::_relationships_inputId);
        attributes.addAttribute(FabricTypes::_relationships_outputId, FabricTokens::_relationships_outputId);
        attributes.addAttribute(FabricTypes::_relationships_inputName, FabricTokens::_relationships_inputName);
        attributes.addAttribute(FabricTypes::_relationships_outputName, FabricTokens::_relationships_outputName);
        attributes.addAttribute(FabricTypes::primvars, FabricTokens::primvars);
        attributes.addAttribute(FabricTypes::MaterialNetwork, FabricTokens::MaterialNetwork);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(displacementPathFabric);

        auto nodePathsCount = 1;
        auto inputsCount = 0;
        auto outputsCount = 0;

        if (hasBaseColorTexture) {
            nodePathsCount = 2;
            inputsCount = 1;
            outputsCount = 1;
        }

        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_nodePaths, nodePathsCount);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_inputId, inputsCount);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_outputId, outputsCount);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_inputName, inputsCount);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_outputName, outputsCount);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::primvars, 0);

        // clang-format off
        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_nodePaths);
        auto relationshipsInputIdFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_relationships_inputId);
        auto relationshipsOutputIdFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_relationships_outputId);
        auto relationshipsInputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(displacementPathFabric, FabricTokens::_relationships_inputName);
        auto relationshipsOutputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(displacementPathFabric, FabricTokens::_relationships_outputName);
        // clang-format on

        if (hasBaseColorTexture) {
            nodePathsFabric[0] = baseColorTexPathFabricUint64;
            nodePathsFabric[1] = shaderPathFabricUint64;
            relationshipsInputIdFabric[0] = baseColorTexPathFabricUint64;
            relationshipsOutputIdFabric[0] = shaderPathFabricUint64;
            relationshipsInputNameFabric[0] = FabricTokens::out;
            relationshipsOutputNameFabric[0] = FabricTokens::base_color_texture;
        } else {
            nodePathsFabric[0] = shaderPathFabricUint64;
        }
    }

    // Surface
    {
        sip.createPrim(surfacePathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::_nodePaths, FabricTokens::_nodePaths);
        attributes.addAttribute(FabricTypes::_relationships_inputId, FabricTokens::_relationships_inputId);
        attributes.addAttribute(FabricTypes::_relationships_outputId, FabricTokens::_relationships_outputId);
        attributes.addAttribute(FabricTypes::_relationships_inputName, FabricTokens::_relationships_inputName);
        attributes.addAttribute(FabricTypes::_relationships_outputName, FabricTokens::_relationships_outputName);
        attributes.addAttribute(FabricTypes::primvars, FabricTokens::primvars);
        attributes.addAttribute(FabricTypes::MaterialNetwork, FabricTokens::MaterialNetwork);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(surfacePathFabric);

        auto nodePathsCount = 1;
        auto inputsCount = 0;
        auto outputsCount = 0;

        if (hasBaseColorTexture) {
            nodePathsCount = 2;
            inputsCount = 1;
            outputsCount = 1;
        }

        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_nodePaths, nodePathsCount);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_inputId, inputsCount);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_outputId, outputsCount);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_inputName, inputsCount);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_outputName, outputsCount);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::primvars, 0);

        // clang-format off
        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_nodePaths);
        auto relationshipsInputIdFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_relationships_inputId);
        auto relationshipsOutputIdFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_relationships_outputId);
        auto relationshipsInputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(surfacePathFabric, FabricTokens::_relationships_inputName);
        auto relationshipsOutputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(surfacePathFabric, FabricTokens::_relationships_outputName);
        // clang-format on

        if (hasBaseColorTexture) {
            nodePathsFabric[0] = baseColorTexPathFabricUint64;
            nodePathsFabric[1] = shaderPathFabricUint64;
            relationshipsInputIdFabric[0] = baseColorTexPathFabricUint64;
            relationshipsOutputIdFabric[0] = shaderPathFabricUint64;
            relationshipsInputNameFabric[0] = FabricTokens::out;
            relationshipsOutputNameFabric[0] = FabricTokens::base_color_texture;
        } else {
            nodePathsFabric[0] = shaderPathFabricUint64;
        }
    }

    // Shader
    {
        sip.createPrim(shaderPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::alpha_cutoff, FabricTokens::alpha_cutoff);
        attributes.addAttribute(FabricTypes::alpha_mode, FabricTokens::alpha_mode);
        attributes.addAttribute(FabricTypes::base_alpha, FabricTokens::base_alpha);
        attributes.addAttribute(FabricTypes::base_color_factor, FabricTokens::base_color_factor);
        attributes.addAttribute(FabricTypes::emissive_factor, FabricTokens::emissive_factor);
        attributes.addAttribute(FabricTypes::metallic_factor, FabricTokens::metallic_factor);
        attributes.addAttribute(FabricTypes::roughness_factor, FabricTokens::roughness_factor);
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        if (hasVertexColors) {
            attributes.addAttribute(FabricTypes::vertex_color_name, FabricTokens::vertex_color_name);
        }

        attributes.createAttributes(shaderPathFabric);

        const size_t parametersCount = hasVertexColors ? 8 : 7;

        sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::_paramColorSpace, 0);
        sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::_parameters, parametersCount);

        // clang-format off
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::_parameters);
        // clang-format on

        *infoIdFabric = FabricTokens::gltf_pbr_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::gltf_material;
        parametersFabric[0] = FabricTokens::alpha_cutoff;
        parametersFabric[1] = FabricTokens::alpha_mode;
        parametersFabric[2] = FabricTokens::base_alpha;
        parametersFabric[3] = FabricTokens::base_color_factor;
        parametersFabric[4] = FabricTokens::emissive_factor;
        parametersFabric[5] = FabricTokens::metallic_factor;
        parametersFabric[6] = FabricTokens::roughness_factor;

        if (hasVertexColors) {
            const auto vertexColorPrimvarNameSize = UsdTokens::vertexColor.GetString().size();
            sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::vertex_color_name, vertexColorPrimvarNameSize);
            auto vertexColorNameFabric =
                sip.getArrayAttributeWr<uint8_t>(shaderPathFabric, FabricTokens::vertex_color_name);
            memcpy(vertexColorNameFabric.data(), UsdTokens::vertexColor.GetText(), vertexColorPrimvarNameSize);
            parametersFabric[7] = FabricTokens::vertex_color_name;
        }
    }

    if (hasBaseColorTexture) {
        // Create the base color texture
        const auto baseColorTextureName =
            fmt::format("{}_base_color_texture", UsdUtil::getSafeName(materialPath.GetString()));
        const auto baseColorTexturePath =
            pxr::SdfAssetPath(fmt::format("{}{}", rtx::resourcemanager::kDynamicTexturePrefix, baseColorTextureName));
        _baseColorTexture = std::make_unique<omni::ui::DynamicTextureProvider>(baseColorTextureName);

        // baseColorTex
        {
            sip.createPrim(baseColorTexPathFabric);

            FabricAttributesBuilder attributes;

            // clang-format off
            attributes.addAttribute(FabricTypes::offset, FabricTokens::offset);
            attributes.addAttribute(FabricTypes::rotation, FabricTokens::rotation);
            attributes.addAttribute(FabricTypes::scale, FabricTokens::scale);
            attributes.addAttribute(FabricTypes::tex_coord_index, FabricTokens::tex_coord_index);
            attributes.addAttribute(FabricTypes::texture, FabricTokens::texture);
            attributes.addAttribute(FabricTypes::wrap_s, FabricTokens::wrap_s);
            attributes.addAttribute(FabricTypes::wrap_t, FabricTokens::wrap_t);
            attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
            attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
            attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
            attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
            attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
            attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
            attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
            // clang-format on

            attributes.createAttributes(baseColorTexPathFabric);

            // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
            sip.setArrayAttributeSize(baseColorTexPathFabric, FabricTokens::_paramColorSpace, 2);
            sip.setArrayAttributeSize(baseColorTexPathFabric, FabricTokens::_parameters, 7);

            // clang-format off
            auto offsetFabric = sip.getAttributeWr<pxr::GfVec2f>(baseColorTexPathFabric, FabricTokens::offset);
            auto rotationFabric = sip.getAttributeWr<float>(baseColorTexPathFabric, FabricTokens::rotation);
            auto scaleFabric = sip.getAttributeWr<pxr::GfVec2f>(baseColorTexPathFabric, FabricTokens::scale);
            auto texCoordIndexFabric = sip.getAttributeWr<int>(baseColorTexPathFabric, FabricTokens::tex_coord_index);
            auto textureFabric = sip.getAttributeWr<FabricAsset>(baseColorTexPathFabric, FabricTokens::texture);
            auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(baseColorTexPathFabric, FabricTokens::info_id);
            auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(baseColorTexPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
            auto paramColorSpaceFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(baseColorTexPathFabric, FabricTokens::_paramColorSpace);
            auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(baseColorTexPathFabric, FabricTokens::_parameters);
            // clang-format on

            *offsetFabric = pxr::GfVec2f(0.0f, 0.0f);
            *rotationFabric = 0.0f;
            *scaleFabric = pxr::GfVec2f(1.0f, 1.0f);
            *texCoordIndexFabric = 0;
            *textureFabric = FabricAsset(baseColorTexturePath);
            *infoIdFabric = FabricTokens::gltf_pbr_mdl;
            *infoSourceAssetSubIdentifierFabric = FabricTokens::gltf_texture_lookup;
            paramColorSpaceFabric[0] = FabricTokens::texture;
            paramColorSpaceFabric[1] = FabricTokens::_auto;
            parametersFabric[0] = FabricTokens::offset;
            parametersFabric[1] = FabricTokens::rotation;
            parametersFabric[2] = FabricTokens::scale;
            parametersFabric[3] = FabricTokens::tex_coord_index;
            parametersFabric[4] = FabricTokens::texture;
            parametersFabric[5] = FabricTokens::wrap_s;
            parametersFabric[6] = FabricTokens::wrap_t;
        }
    }

    _materialPathFabric = materialPathFabric;
    _shaderPathFabric = shaderPathFabric;
    _displacementPathFabric = displacementPathFabric;
    _surfacePathFabric = surfacePathFabric;
    _baseColorTexPathFabric = baseColorTexPathFabric;

    reset();
}

void FabricMaterial::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    auto sip = UsdUtil::getFabricStageInProgress();

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
    auto alphaCutoffFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::alpha_cutoff);
    auto alphaModeFabric = sip.getAttributeWr<int>(_shaderPathFabric, FabricTokens::alpha_mode);
    auto baseAlphaFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::base_alpha);
    auto baseColorFactorFabric = sip.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::base_color_factor);
    auto emissiveFactorFabric = sip.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::emissive_factor);
    auto metallicFactorFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::metallic_factor);
    auto roughnessFactorFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::roughness_factor);
    // clang-format on

    *alphaCutoffFabric = alphaCutoff;
    *alphaModeFabric = alphaMode;
    *baseAlphaFabric = baseAlpha;
    *baseColorFactorFabric = baseColorFactor;
    *emissiveFactorFabric = emissiveFactor;
    *metallicFactorFabric = metallicFactor;
    *roughnessFactorFabric = roughnessFactor;

    if (hasBaseColorTexture) {
        auto wrapSFabric = sip.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::wrap_s);
        auto wrapTFabric = sip.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::wrap_t);

        *wrapSFabric = baseColorTextureWrapS;
        *wrapTFabric = baseColorTextureWrapT;
    }

    FabricUtil::setTilesetIdAndTileId(_materialPathFabric, -1, -1);
    FabricUtil::setTilesetIdAndTileId(_shaderPathFabric, -1, -1);
    FabricUtil::setTilesetIdAndTileId(_displacementPathFabric, -1, -1);
    FabricUtil::setTilesetIdAndTileId(_surfacePathFabric, -1, -1);

    if (hasBaseColorTexture) {
        FabricUtil::setTilesetIdAndTileId(_baseColorTexPathFabric, -1, -1);
    }
}

void FabricMaterial::setInitialValues(const FabricMaterialDefinition& materialDefinition) {
    const auto hasBaseColorTexture = _materialDefinition.hasBaseColorTexture();

    auto sip = UsdUtil::getFabricStageInProgress();

    // clang-format off
    auto alphaCutoffFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::alpha_cutoff);
    auto alphaModeFabric = sip.getAttributeWr<int>(_shaderPathFabric, FabricTokens::alpha_mode);
    auto baseAlphaFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::base_alpha);
    auto baseColorFactorFabric = sip.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::base_color_factor);
    auto emissiveFactorFabric = sip.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::emissive_factor);
    auto metallicFactorFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::metallic_factor);
    auto roughnessFactorFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::roughness_factor);
    // clang-format on

    *alphaCutoffFabric = materialDefinition.getAlphaCutoff();
    *alphaModeFabric = materialDefinition.getAlphaMode();
    *baseAlphaFabric = materialDefinition.getBaseAlpha();
    *baseColorFactorFabric = materialDefinition.getBaseColorFactor();
    *emissiveFactorFabric = materialDefinition.getEmissiveFactor();
    *metallicFactorFabric = materialDefinition.getMetallicFactor();
    *roughnessFactorFabric = materialDefinition.getRoughnessFactor();

    if (hasBaseColorTexture) {
        auto wrapSFabric = sip.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::wrap_s);
        auto wrapTFabric = sip.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::wrap_t);

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

    auto sip = UsdUtil::getFabricStageInProgress();

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
    auto alphaCutoffFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::alpha_cutoff);
    auto alphaModeFabric = sip.getAttributeWr<int>(_shaderPathFabric, FabricTokens::alpha_mode);
    auto baseAlphaFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::base_alpha);
    auto baseColorFactorFabric = sip.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::base_color_factor);
    auto emissiveFactorFabric = sip.getAttributeWr<pxr::GfVec3f>(_shaderPathFabric, FabricTokens::emissive_factor);
    auto metallicFactorFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::metallic_factor);
    auto roughnessFactorFabric = sip.getAttributeWr<float>(_shaderPathFabric, FabricTokens::roughness_factor);
    // clang-format on

    *alphaCutoffFabric = alphaCutoff;
    *alphaModeFabric = alphaMode;
    *baseAlphaFabric = baseAlpha;
    *baseColorFactorFabric = baseColorFactor;
    *emissiveFactorFabric = emissiveFactor;
    *metallicFactorFabric = metallicFactor;
    *roughnessFactorFabric = roughnessFactor;

    if (hasBaseColorTexture) {
        auto wrapSFabric = sip.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::wrap_s);
        auto wrapTFabric = sip.getAttributeWr<int>(_baseColorTexPathFabric, FabricTokens::wrap_t);

        *wrapSFabric = baseColorTextureWrapS;
        *wrapTFabric = baseColorTextureWrapT;
    }

    FabricUtil::setTilesetIdAndTileId(_materialPathFabric, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_shaderPathFabric, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_displacementPathFabric, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_surfacePathFabric, tilesetId, tileId);

    if (hasBaseColorTexture) {
        FabricUtil::setTilesetIdAndTileId(_baseColorTexPathFabric, tilesetId, tileId);
    }
}
}; // namespace cesium::omniverse
