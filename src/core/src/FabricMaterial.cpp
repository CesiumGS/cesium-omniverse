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
    : _materialDefinition(materialDefinition)
    , _materialPath(path) {

    initialize(path, materialDefinition);

    // Remove this function once dynamic material values are supported in Kit 105
    setInitialValues(materialDefinition);
}

FabricMaterial::~FabricMaterial() {
    FabricUtil::destroyPrim(_materialPath);
    FabricUtil::destroyPrim(_shaderPath);
    FabricUtil::destroyPrim(_lookupColorPath);
    FabricUtil::destroyPrim(_textureCoordinate2dPath);
}

void FabricMaterial::setActive(bool active) {
    if (!active) {
        reset();
    }
}

pxr::SdfPath FabricMaterial::getPath() const {
    return _materialPath;
}

const FabricMaterialDefinition& FabricMaterial::getMaterialDefinition() const {
    return _materialDefinition;
}

void FabricMaterial::initialize(pxr::SdfPath path, const FabricMaterialDefinition& materialDefinition) {
    const auto hasBaseColorTexture = materialDefinition.hasBaseColorTexture();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto materialPath = path;
    const auto shaderPath = materialPath.AppendChild(UsdTokens::Shader);
    const auto lookupColorPath = materialPath.AppendChild(UsdTokens::lookup_color);
    const auto textureCoordinate2dPath = materialPath.AppendChild(UsdTokens::texture_coordinate_2d);

    const auto shaderPathFabricUint64 = omni::fabric::asInt(shaderPath).path;
    const auto lookupColorPathFabricUint64 = omni::fabric::asInt(lookupColorPath).path;
    const auto textureCoordinate2dPathFabricUint64 = omni::fabric::asInt(textureCoordinate2dPath).path;

    // Material
    {
        const auto materialPathFabric = omni::fabric::Path(omni::fabric::asInt(materialPath));
        srw.createPrim(materialPathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
        attributes.addAttribute(FabricTypes::_nodePaths, FabricTokens::_nodePaths);
        attributes.addAttribute(FabricTypes::_terminal_names, FabricTokens::_terminal_names);
        attributes.addAttribute(FabricTypes::_terminal_sourceNames, FabricTokens::_terminal_sourceNames);
        attributes.addAttribute(FabricTypes::_terminal_sourceIds, FabricTokens::_terminal_sourceIds);
        attributes.addAttribute(FabricTypes::_relationship_ids, FabricTokens::_relationship_ids);
        attributes.addAttribute(FabricTypes::_relationship_names, FabricTokens::_relationship_names);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(materialPathFabric);

        auto nodePathsCount = 1;
        auto relationshipCount = 0;

        if (hasBaseColorTexture) {
            nodePathsCount = 3;
            relationshipCount = 4;
        }

        srw.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminal_names, 3);
        srw.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminal_sourceNames, 3);
        srw.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminal_sourceIds, 3);
        srw.setArrayAttributeSize(materialPathFabric, FabricTokens::_nodePaths, nodePathsCount);
        srw.setArrayAttributeSize(materialPathFabric, FabricTokens::_relationship_ids, relationshipCount);
        srw.setArrayAttributeSize(materialPathFabric, FabricTokens::_relationship_names, relationshipCount);

        // clang-format off
        auto terminalNamesFabric = srw.getArrayAttributeWr<omni::fabric::Token>(materialPathFabric, FabricTokens::_terminal_names);
        auto terminalSourceNamesFabric = srw.getArrayAttributeWr<omni::fabric::Token>(materialPathFabric, FabricTokens::_terminal_sourceNames);
        auto terminalSourceIdsFabric = srw.getArrayAttributeWr<int>(materialPathFabric, FabricTokens::_terminal_sourceIds);
        auto nodePathsFabric = srw.getArrayAttributeWr<uint64_t>(materialPathFabric, FabricTokens::_nodePaths);
        auto relationshipIdsFabric = srw.getArrayAttributeWr<int>(materialPathFabric, FabricTokens::_relationship_ids);
        auto relationshipNamesFabric = srw.getArrayAttributeWr<omni::fabric::Token>(materialPathFabric, FabricTokens::_relationship_names);
        // clang-format on

        terminalNamesFabric[0] = FabricTokens::outputs_mdl_displacement;
        terminalNamesFabric[1] = FabricTokens::outputs_mdl_surface;
        terminalNamesFabric[2] = FabricTokens::outputs_mdl_volume;

        terminalSourceNamesFabric[0] = FabricTokens::out;
        terminalSourceNamesFabric[1] = FabricTokens::out;
        terminalSourceNamesFabric[2] = FabricTokens::out;

        terminalSourceIdsFabric[0] = 0;
        terminalSourceIdsFabric[1] = 0;
        terminalSourceIdsFabric[2] = 0;

        if (hasBaseColorTexture) {
            nodePathsFabric[0] = shaderPathFabricUint64;
            nodePathsFabric[1] = lookupColorPathFabricUint64;
            nodePathsFabric[2] = textureCoordinate2dPathFabricUint64;

            relationshipIdsFabric[0] = 2; // texture coordinate
            relationshipIdsFabric[1] = 1; // lookup color
            relationshipIdsFabric[2] = 1; // lookup color
            relationshipIdsFabric[3] = 0; // shader

            relationshipNamesFabric[0] = FabricTokens::outputs_out;
            relationshipNamesFabric[1] = FabricTokens::inputs_coord;
            relationshipNamesFabric[2] = FabricTokens::outputs_out;
            relationshipNamesFabric[3] = FabricTokens::inputs_diffuse_color_constant;

        } else {
            nodePathsFabric[0] = shaderPathFabricUint64;
        }
    }

    auto addShaderAttributes = [](FabricAttributesBuilder& attributes) {
        // clang-format off
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::info_implementationSource, FabricTokens::info_implementationSource);
        attributes.addAttribute(FabricTypes::info_mdl_sourceAsset, FabricTokens::info_mdl_sourceAsset);
        attributes.addAttribute(FabricTypes::info_mdl_sourceAsset_subIdentifier, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_sdrMetadata, FabricTokens::_sdrMetadata);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on
    };

    auto setShaderAttributes = [&srw](const omni::fabric::Path& shaderPathArg, bool setColorSpaceEmpty = true) {
        auto infoImplementationSourceFabric =
            srw.getAttributeWr<omni::fabric::Token>(shaderPathArg, FabricTokens::info_implementationSource);

        *infoImplementationSourceFabric = FabricTokens::sourceAsset;

        // _sdrMetadata is used with the sdr registry calls.
        srw.setArrayAttributeSize(shaderPathArg, FabricTokens::_sdrMetadata, 0);

        if (setColorSpaceEmpty) {
            // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
            srw.setArrayAttributeSize(shaderPathArg, FabricTokens::_paramColorSpace, 0);
        }
    };

    // Shader
    {
        const auto shaderPathFabric = omni::fabric::Path(omni::fabric::asInt(shaderPath));
        srw.createPrim(shaderPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        addShaderAttributes(attributes);
        attributes.addAttribute(FabricTypes::inputs_diffuse_color_constant, FabricTokens::inputs_diffuse_color_constant);
        attributes.addAttribute(FabricTypes::inputs_metallic_constant, FabricTokens::inputs_metallic_constant);
        attributes.addAttribute(FabricTypes::inputs_reflection_roughness_constant, FabricTokens::inputs_reflection_roughness_constant);
        attributes.addAttribute(FabricTypes::inputs_specular_level, FabricTokens::inputs_specular_level);
        // clang-format on

        attributes.createAttributes(shaderPathFabric);

        setShaderAttributes(shaderPathFabric);

        // clang-format off
        auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(shaderPathFabric, FabricTokens::info_mdl_sourceAsset);
        auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(shaderPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        auto specularLevelFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_specular_level);
        // clang-format on

        infoMdlSourceAssetFabric->assetPath = UsdTokens::OmniPBR_mdl;
        infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
        *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::OmniPBR;
        *specularLevelFabric = 0.0f;
    }

    if (hasBaseColorTexture) {

        // Create the base color texture
        const auto baseColorTextureName =
            fmt::format("{}_base_color_texture", UsdUtil::getSafeName(materialPath.GetString()));
        const auto baseColorTexturePath =
            pxr::SdfAssetPath(fmt::format("{}{}", rtx::resourcemanager::kDynamicTexturePrefix, baseColorTextureName));
        _baseColorTexture = std::make_unique<omni::ui::DynamicTextureProvider>(baseColorTextureName);

        // texture_coordinate_2d
        {
            const auto textureCoordinate2dPathFabric = omni::fabric::Path(omni::fabric::asInt(textureCoordinate2dPath));
            srw.createPrim(textureCoordinate2dPathFabric);

            FabricAttributesBuilder attributes;

            addShaderAttributes(attributes);

            attributes.createAttributes(textureCoordinate2dPathFabric);

            setShaderAttributes(textureCoordinate2dPathFabric);

            // clang-format off
            auto infoMdlSourceAssetFabric = srw.getAttributeWr<omni::fabric::AssetPath>(textureCoordinate2dPathFabric, FabricTokens::info_mdl_sourceAsset);
            auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(textureCoordinate2dPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            // clang-format on

            infoMdlSourceAssetFabric->assetPath = UsdTokens::nvidia_support_definitions_mdl;
            infoMdlSourceAssetFabric->resolvedPath = pxr::TfToken();
            *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::texture_coordinate_2d;
        }

        // lookup_color
        {
            const auto lookupColorPathFabric = omni::fabric::Path(omni::fabric::asInt(lookupColorPath));
            srw.createPrim(lookupColorPathFabric);

            FabricAttributesBuilder attributes;

            // clang-format off
            addShaderAttributes(attributes);
            attributes.addAttribute(FabricTypes::inputs_tex, FabricTokens::inputs_tex);
            attributes.addAttribute(FabricTypes::inputs_coord, FabricTokens::inputs_coord);
            attributes.addAttribute(FabricTypes::inputs_wrap_u, FabricTokens::inputs_wrap_u);
            attributes.addAttribute(FabricTypes::inputs_wrap_v, FabricTokens::inputs_wrap_v);
            // clang-format on

            attributes.createAttributes(lookupColorPathFabric);

            setShaderAttributes(lookupColorPathFabric, false);

            srw.setArrayAttributeSize(lookupColorPathFabric, FabricTokens::_paramColorSpace, 2);

            // clang-format off
            auto wrapUFabric = srw.getAttributeWr<int>(lookupColorPathFabric, FabricTokens::inputs_wrap_u);
            auto wrapVFabric = srw.getAttributeWr<int>(lookupColorPathFabric, FabricTokens::inputs_wrap_v);
            auto coordFabric = srw.getAttributeWr<pxr::GfVec2f>(lookupColorPathFabric, FabricTokens::inputs_coord);
            auto texFabric = srw.getAttributeWr<omni::fabric::AssetPath>(lookupColorPathFabric, FabricTokens::inputs_tex);
            auto infoMdlSourceAsset = srw.getAttributeWr<omni::fabric::AssetPath>(lookupColorPathFabric, FabricTokens::info_mdl_sourceAsset);
            auto infoMdlSourceAssetSubIdentifierFabric = srw.getAttributeWr<omni::fabric::Token>(lookupColorPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            auto paramColorSpaceFabric = srw.getArrayAttributeWr<omni::fabric::Token>(lookupColorPathFabric, FabricTokens::_paramColorSpace);
            // clang-format on

            *wrapUFabric = 0; // clamp to edge
            *wrapVFabric = 0; // clamp to edge
            *coordFabric = pxr::GfVec2f(0.0f);
            texFabric->assetPath = pxr::TfToken(baseColorTexturePath.GetAssetPath());
            texFabric->resolvedPath = pxr::TfToken(baseColorTexturePath.GetResolvedPath());
            infoMdlSourceAsset->assetPath = UsdTokens::nvidia_support_definitions_mdl;
            infoMdlSourceAsset->resolvedPath = pxr::TfToken();
            *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::lookup_color;
            paramColorSpaceFabric[0] = FabricTokens::inputs_tex;
            paramColorSpaceFabric[1] = FabricTokens::_auto;
        }
    }

    _materialPath = materialPath;
    _shaderPath = shaderPath;
    _lookupColorPath = lookupColorPath;
    _textureCoordinate2dPath = textureCoordinate2dPath;

    reset();
}

void FabricMaterial::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();

    if (_baseColorTexture != nullptr) {
        // Clear the texture
        const auto bytes = std::array<uint8_t, 4>{{255, 255, 255, 255}};
        const auto size = carb::Uint2{1, 1};
        _baseColorTexture->setBytesData(bytes.data(), size, omni::ui::kAutoCalculateStride, carb::Format::eRGBA8_SRGB);
    }

    const auto baseColorFactor = GltfUtil::getDefaultBaseColorFactor();
    const auto metallicFactor = GltfUtil::getDefaultMetallicFactor();
    const auto roughnessFactor = GltfUtil::getDefaultRoughnessFactor();

    const auto shaderPathFabric = omni::fabric::Path(omni::fabric::asInt(_shaderPath));

    // clang-format off
    auto diffuseColorConstantFabric = srw.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::inputs_diffuse_color_constant);
    auto metallicConstantFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_metallic_constant);
    auto reflectionRoughnessConstantFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_reflection_roughness_constant);
    // clang-format on

    *diffuseColorConstantFabric = baseColorFactor;
    *metallicConstantFabric = metallicFactor;
    *reflectionRoughnessConstantFabric = roughnessFactor;

    FabricUtil::setTilesetIdAndTileId(_materialPath, -1, -1);
    FabricUtil::setTilesetIdAndTileId(_shaderPath, -1, -1);
}

void FabricMaterial::setInitialValues(const FabricMaterialDefinition& materialDefinition) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto shaderPathFabric = omni::fabric::Path(omni::fabric::asInt(_shaderPath));

    // clang-format off
    auto diffuseColorConstantFabric = srw.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::inputs_diffuse_color_constant);
    auto metallicConstantFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_metallic_constant);
    auto reflectionRoughnessConstantFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_reflection_roughness_constant);
    // clang-format on

    *diffuseColorConstantFabric = materialDefinition.getBaseColorFactor();
    *metallicConstantFabric = materialDefinition.getMetallicFactor();
    *reflectionRoughnessConstantFabric = materialDefinition.getRoughnessFactor();
}

void FabricMaterial::setTile(
    int64_t tilesetId,
    int64_t tileId,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const CesiumGltf::ImageCesium* imagery) {

    auto srw = UsdUtil::getFabricStageReaderWriter();

    pxr::GfVec3f baseColorFactor;
    float metallicFactor;
    float roughnessFactor;

    const auto hasGltfMaterial = GltfUtil::hasMaterial(primitive);

    const CesiumGltf::ImageCesium* baseColorImage = nullptr;

    if (hasGltfMaterial) {
        const auto& material = model.materials[static_cast<size_t>(primitive.material)];

        baseColorFactor = GltfUtil::getBaseColorFactor(material);
        metallicFactor = GltfUtil::getMetallicFactor(material);
        roughnessFactor = GltfUtil::getRoughnessFactor(material);

        const auto baseColorTextureIndex = GltfUtil::getBaseColorTextureIndex(model, material);

        if (baseColorTextureIndex.has_value()) {
            baseColorImage = &GltfUtil::getImageCesium(model, model.textures[baseColorTextureIndex.value()]);
        }
    } else {
        baseColorFactor = GltfUtil::getDefaultBaseColorFactor();
        metallicFactor = GltfUtil::getDefaultMetallicFactor();
        roughnessFactor = GltfUtil::getDefaultRoughnessFactor();
    }

    // Imagery overrides the base color texture in the glTF
    if (imagery != nullptr) {
        baseColorImage = imagery;
    }

    if (baseColorImage != nullptr) {
        _baseColorTexture->setBytesData(
            reinterpret_cast<const uint8_t*>(baseColorImage->pixelData.data()),
            carb::Uint2{static_cast<uint32_t>(baseColorImage->width), static_cast<uint32_t>(baseColorImage->height)},
            omni::ui::kAutoCalculateStride,
            carb::Format::eRGBA8_SRGB);
    }

    const auto shaderPathFabric = omni::fabric::Path(omni::fabric::asInt(_shaderPath));

    // clang-format off
    auto diffuseColorConstantFabric = srw.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::inputs_diffuse_color_constant);
    auto metallicConstantFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_metallic_constant);
    auto reflectionRoughnessConstantFabric = srw.getAttributeWr<float>(shaderPathFabric, FabricTokens::inputs_reflection_roughness_constant);
    // clang-format on

    *diffuseColorConstantFabric = baseColorFactor;
    *metallicConstantFabric = metallicFactor;
    *reflectionRoughnessConstantFabric = roughnessFactor;

    FabricUtil::setTilesetIdAndTileId(_materialPath, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_shaderPath, tilesetId, tileId);
}
}; // namespace cesium::omniverse
