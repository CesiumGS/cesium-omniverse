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
    FabricUtil::destroyPrim(_displacementPath);
    FabricUtil::destroyPrim(_surfacePath);
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

    auto sip = UsdUtil::getFabricStageReaderWriter();
    auto isip = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();

    const auto materialPath = path;
    const auto shaderPath = materialPath.AppendChild(UsdTokens::Shader);
    const auto displacementPath = materialPath.AppendChild(UsdTokens::displacement);
    const auto surfacePath = materialPath.AppendChild(UsdTokens::surface);
    const auto lookupColorPath = materialPath.AppendChild(UsdTokens::lookup_color);
    const auto textureCoordinate2dPath = materialPath.AppendChild(UsdTokens::texture_coordinate_2d);

    const auto shaderPathFabricUint64 = omni::fabric::PathC(omni::fabric::asInt(shaderPath)).path;
    const auto lookupColorPathFabricUint64 = omni::fabric::PathC(omni::fabric::asInt(lookupColorPath)).path;
    const auto textureCoordinate2dPathFabricUint64 = omni::fabric::PathC(omni::fabric::asInt(textureCoordinate2dPath)).path;

    // Material
    {
        const auto materialPathFabric = omni::fabric::Path(omni::fabric::asInt(materialPath));
        sip.createPrim(materialPathFabric);

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

        // TODO(jshrake): Temporarily disable the lookup node and set inputs:diffuse_texture directly on the MDL
        if (false && hasBaseColorTexture) {
            nodePathsCount = 3;
            relationshipCount = 2;
        }

        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminal_names, 2);
        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminal_sourceNames, 2);
        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminal_sourceIds, 2);
        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_nodePaths, nodePathsCount);
        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_relationship_ids, relationshipCount * 2);
        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_relationship_names, relationshipCount * 2);

        auto terminalNamesFabric = sip.getArrayAttributeWr<omni::fabric::Token>(materialPathFabric, FabricTokens::_terminal_names);
        auto terminalSourceNamesFabric =
            sip.getArrayAttributeWr<omni::fabric::Token>(materialPathFabric, FabricTokens::_terminal_sourceNames);
        auto terminalSourceIdsFabric =
            sip.getArrayAttributeWr<int>(materialPathFabric, FabricTokens::_terminal_sourceIds);

        terminalNamesFabric[0] = FabricTokens::outputs_surface;
        terminalSourceNamesFabric[0] = FabricTokens::outputs_out;
        terminalSourceIdsFabric[0] = 0;
        terminalNamesFabric[1] = FabricTokens::outputs_displacement;
        terminalSourceNamesFabric[1] = FabricTokens::outputs_out;
        terminalSourceIdsFabric[1] = 0;

        // clang-format off
        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(materialPathFabric, FabricTokens::_nodePaths);
        auto relationshipIdsFabric = sip.getArrayAttributeWr<int>(materialPathFabric, FabricTokens::_relationship_ids);
        auto relationshipNamesFabric = sip.getArrayAttributeWr<omni::fabric::Token>(materialPathFabric, FabricTokens::_relationship_names);
        // clang-format on

        // TODO(jshrake): Temporarily disable the lookup node and set inputs:diffuse_texture directly on the MDL
        if (false && hasBaseColorTexture) {
            nodePathsFabric[0] = shaderPathFabricUint64;
            nodePathsFabric[1] = lookupColorPathFabricUint64;
            nodePathsFabric[2] = textureCoordinate2dPathFabricUint64;
            // Indices into the nodePaths array above.
            relationshipIdsFabric[0] = 2; // texture coordinate
            relationshipNamesFabric[0] = FabricTokens::out;
            relationshipIdsFabric[1] = 1; // lookup color
            relationshipNamesFabric[1] = FabricTokens::coord;
            relationshipIdsFabric[2] = 1; // lookup color
            relationshipNamesFabric[2] = FabricTokens::out;
            relationshipIdsFabric[3] = 0; // shader
            relationshipNamesFabric[3] = FabricTokens::diffuse_color_constant;
        } else {
            nodePathsFabric[0] = shaderPathFabricUint64;
        }
    }

    auto addShaderParams = [](FabricAttributesBuilder& attributes) {
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::info_implementationSource, FabricTokens::info_implementationSource);
        attributes.addAttribute(
            FabricTypes::info_mdl_sourceAsset, FabricTokens::info_mdl_sourceAsset);
        attributes.addAttribute(
            FabricTypes::info_mdl_sourceAsset_subIdentifier, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_sdrMetadata, FabricTokens::_sdrMetadata);
    };

    auto setShaderParams = [&sip](const omni::fabric::Path& shaderPath, bool setColorSpaceEmpty = true) {
        *sip.getAttributeWr<omni::fabric::Token>(shaderPath, FabricTokens::info_implementationSource) =
            FabricTokens::sourceAsset;
        // _sdrMetadata is used with the sdr registry calls.
        sip.setArrayAttributeSize(shaderPath, FabricTokens::_sdrMetadata, 0);
        if (setColorSpaceEmpty)
        {
            // _paramColorSpace is an array of pairs: [texture_parameter_token, color_space_enum], [texture_parameter_token, color_space_enum], ...
            sip.setArrayAttributeSize(shaderPath, FabricTokens::_paramColorSpace, 0);
        }
    };

    // Shader
    {
        const auto shaderPathFabric = omni::fabric::Path(omni::fabric::asInt(shaderPath));
        sip.createPrim(shaderPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        addShaderParams(attributes);
        attributes.addAttribute(FabricTypes::diffuse_texture, FabricTokens::diffuse_texture);
        attributes.addAttribute(FabricTypes::diffuse_color_constant, FabricTokens::diffuse_color_constant);
        attributes.addAttribute(FabricTypes::metallic_constant, FabricTokens::metallic_constant);
        attributes.addAttribute(FabricTypes::reflection_roughness_constant, FabricTokens::reflection_roughness_constant);
        attributes.addAttribute(FabricTypes::specular_level, FabricTokens::specular_level);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        attributes.createAttributes(shaderPathFabric);

        setShaderParams(shaderPathFabric);

        // clang-format off
        auto infoMdlSourceAssetSpan = isip->getAttributeWr(sip.getId(), shaderPathFabric, FabricTokens::info_mdl_sourceAsset);
        auto* infoMdlSourceAsset = reinterpret_cast<omni::fabric::AssetPath*>(infoMdlSourceAssetSpan.ptr);
        auto infoMdlSourceAssetSubIdentifierFabric = sip.getAttributeWr<omni::fabric::Token>(shaderPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        auto specularLevelFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::specular_level);
        // clang-format on

        infoMdlSourceAsset->assetPath = UsdTokens::OmniPBR_mdl;
        infoMdlSourceAsset->resolvedPath = pxr::TfToken();
        *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::OmniPBR;
        *specularLevelFabric = 0.0f;

        // TODO(jshrake): Temporarily disable the lookup node and set inputs:diffuse_texture directly on the MDL
        if (hasBaseColorTexture) {
            const auto baseColorTextureName =
                fmt::format("{}_base_color_texture", UsdUtil::getSafeName(materialPath.GetString()));
            const auto baseColorTexturePath =
                pxr::SdfAssetPath(fmt::format("{}{}", rtx::resourcemanager::kDynamicTexturePrefix, baseColorTextureName));
            auto texSpan = isip->getAttributeWr(sip.getId(), shaderPathFabric, FabricTokens::diffuse_texture);
            auto* diffuseTextureFabric = reinterpret_cast<omni::fabric::AssetPath*>(texSpan.ptr);
            diffuseTextureFabric->assetPath = pxr::TfToken(baseColorTexturePath.GetAssetPath());
            diffuseTextureFabric->resolvedPath = pxr::TfToken(baseColorTexturePath.GetResolvedPath());
            _baseColorTexture = std::make_unique<omni::ui::DynamicTextureProvider>(baseColorTextureName);
        }
    }

    // TODO(jshrake): Temporarily disable the lookup node and set inputs:diffuse_texture directly on the MDL
    if (false && hasBaseColorTexture) {
        // Create the base color texture
        const auto baseColorTextureName =
            fmt::format("{}_base_color_texture", UsdUtil::getSafeName(materialPath.GetString()));
        const auto baseColorTexturePath =
            pxr::SdfAssetPath(fmt::format("{}{}", rtx::resourcemanager::kDynamicTexturePrefix, baseColorTextureName));
        _baseColorTexture = std::make_unique<omni::ui::DynamicTextureProvider>(baseColorTextureName);

        // texture_coordinate_2d
        {
            const auto textureCoordinate2dPathFabric =
                omni::fabric::Path(omni::fabric::asInt(textureCoordinate2dPath));
            sip.createPrim(textureCoordinate2dPathFabric);

            FabricAttributesBuilder attributes;

            // clang-format off
            addShaderParams(attributes);
            attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
            attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
            // clang-format on

            attributes.createAttributes(textureCoordinate2dPathFabric);

            setShaderParams(textureCoordinate2dPathFabric);

            // clang-format off
            auto infoMdlSourceAssetSpan = isip->getAttributeWr(sip.getId(), textureCoordinate2dPathFabric, FabricTokens::info_mdl_sourceAsset);
            auto* infoMdlSourceAsset = reinterpret_cast<omni::fabric::AssetPath*>(infoMdlSourceAssetSpan.ptr);
            auto infoMdlSourceAssetSubIdentifierFabric = sip.getAttributeWr<omni::fabric::Token>(textureCoordinate2dPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            // clang-format on

            infoMdlSourceAsset->assetPath = UsdTokens::nvidia_support_definitions_mdl;
            infoMdlSourceAsset->resolvedPath = pxr::TfToken();
            *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::texture_coordinate_2d;
        }

        // lookup_color
        {
            const auto lookupColorPathFabric = omni::fabric::Path(omni::fabric::asInt(lookupColorPath));
            sip.createPrim(lookupColorPathFabric);

            FabricAttributesBuilder attributes;

            // clang-format off
            addShaderParams(attributes);
            attributes.addAttribute(FabricTypes::wrap_u, FabricTokens::wrap_u);
            attributes.addAttribute(FabricTypes::wrap_v, FabricTokens::wrap_v);
            attributes.addAttribute(FabricTypes::tex, FabricTokens::tex);
            attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
            attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
            // clang-format on

            attributes.createAttributes(lookupColorPathFabric);

            setShaderParams(lookupColorPathFabric, false);

            sip.setArrayAttributeSize(lookupColorPathFabric, FabricTokens::_paramColorSpace, 2);

            // clang-format off
            auto wrapUFabric = sip.getAttributeWr<int>(lookupColorPathFabric, FabricTokens::wrap_u);
            auto wrapVFabric = sip.getAttributeWr<int>(lookupColorPathFabric, FabricTokens::wrap_v);
            auto texFabricSpan = isip->getAttributeWr(sip.getId(), lookupColorPathFabric, FabricTokens::tex);
            auto* texFabric = reinterpret_cast<omni::fabric::AssetPath*>(texFabricSpan.ptr);
            auto infoMdlSourceAssetSpan = isip->getAttributeWr(sip.getId(), lookupColorPathFabric, FabricTokens::tex);
            auto* infoMdlSourceAsset = reinterpret_cast<omni::fabric::AssetPath*>(infoMdlSourceAssetSpan.ptr);
            auto infoMdlSourceAssetSubIdentifierFabric = sip.getAttributeWr<omni::fabric::Token>(lookupColorPathFabric, FabricTokens::info_mdl_sourceAsset_subIdentifier);
            auto paramColorSpaceFabric = sip.getArrayAttributeWr<omni::fabric::Token>(lookupColorPathFabric, FabricTokens::_paramColorSpace);
            // clang-format on

            *wrapUFabric = 0; // clamp to edge
            *wrapVFabric = 0; // clamp to edge
            texFabric->assetPath = pxr::TfToken(baseColorTexturePath.GetAssetPath());
            texFabric->resolvedPath = pxr::TfToken(baseColorTexturePath.GetResolvedPath());
            infoMdlSourceAsset->assetPath = UsdTokens::nvidia_support_definitions_mdl;
            infoMdlSourceAsset->resolvedPath = pxr::TfToken();
            *infoMdlSourceAssetSubIdentifierFabric = FabricTokens::lookup_color;
            paramColorSpaceFabric[0] = FabricTokens::tex;
            paramColorSpaceFabric[1] = FabricTokens::_auto;
        }
    }

    _materialPath = materialPath;
    _shaderPath = shaderPath;
    _displacementPath = displacementPath;
    _surfacePath = surfacePath;

    reset();
}

void FabricMaterial::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    auto sip = UsdUtil::getFabricStageReaderWriter();

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
    auto diffuseColorConstantFabric = sip.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::diffuse_color_constant);
    auto metallicConstantFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::metallic_constant);
    auto reflectionRoughnessConstantFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::reflection_roughness_constant);
    // clang-format on

    *diffuseColorConstantFabric = baseColorFactor;
    *metallicConstantFabric = metallicFactor;
    *reflectionRoughnessConstantFabric = roughnessFactor;

    FabricUtil::setTilesetIdAndTileId(_materialPath, -1, -1);
    FabricUtil::setTilesetIdAndTileId(_shaderPath, -1, -1);
}

void FabricMaterial::setInitialValues(const FabricMaterialDefinition& materialDefinition) {
    auto sip = UsdUtil::getFabricStageReaderWriter();

    const auto shaderPathFabric = omni::fabric::Path(omni::fabric::asInt(_shaderPath));

    // clang-format off
    auto diffuseColorConstantFabric = sip.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::diffuse_color_constant);
    auto metallicConstantFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::metallic_constant);
    auto reflectionRoughnessConstantFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::reflection_roughness_constant);
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

    auto sip = UsdUtil::getFabricStageReaderWriter();

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
    auto diffuseColorConstantFabric = sip.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::diffuse_color_constant);
    auto metallicConstantFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::metallic_constant);
    auto reflectionRoughnessConstantFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::reflection_roughness_constant);
    // clang-format on

    *diffuseColorConstantFabric = baseColorFactor;
    *metallicConstantFabric = metallicFactor;
    *reflectionRoughnessConstantFabric = roughnessFactor;

    FabricUtil::setTilesetIdAndTileId(_materialPath, tilesetId, tileId);
    FabricUtil::setTilesetIdAndTileId(_shaderPath, tilesetId, tileId);
}
}; // namespace cesium::omniverse
