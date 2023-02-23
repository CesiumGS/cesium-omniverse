#include "cesium/omniverse/FabricStageUtil.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/FabricAsset.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/InMemoryAssetResolver.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/GltfUtilities.h>
#include <carb/flatcache/FlatCacheUSD.h>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/usd/sdf/assetPath.h>
#include <spdlog/fmt/fmt.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <regex>

namespace cesium::omniverse::FabricStageUtil {

namespace {

const auto TEXTURE_LOADING_COLOR = pxr::GfVec3f(0.0f, 1.0f, 0.0f);
const auto MATERIAL_LOADING_COLOR = pxr::GfVec3f(1.0f, 0.0f, 0.0f);
const auto DEFAULT_COLOR = pxr::GfVec3f(1.0f, 1.0f, 1.0f);

bool disableMaterials() {
    return Context::instance().getDebugDisableMaterials();
}

std::string getTileNameUnique(int64_t tilesetId, int64_t tileId) {
    return fmt::format("context_{}_tileset_{}_tile_{}", Context::instance().getContextId(), tilesetId, tileId);
}

struct AssetPath {
    std::string assetName;
    pxr::SdfAssetPath assetPath;
};

AssetPath getAssetPath(const std::string& name, const std::string& extension) {
    const auto assetName = fmt::format("{}.{}", name, extension);

    const auto memCesiumPath = Context::instance().getMemCesiumPath().generic_string();
    const auto texturePath = fmt::format("{}[{}]", memCesiumPath, assetName);

    // Set both the path and resolvedPath so that it works with FabricAsset
    const auto assetPath = pxr::SdfAssetPath(texturePath, texturePath);

    return AssetPath{assetName, assetPath};
}

AssetPath getTextureAssetPath(const CesiumGltf::Model& model, int64_t tilesetId, int64_t tileId, uint64_t textureId) {
    std::string texturePrefix;
    const auto urlIt = model.extras.find("Cesium3DTiles_TileUrl");
    if (urlIt != model.extras.end()) {
        texturePrefix = urlIt->second.getStringOrDefault("");
    }

    if (texturePrefix.empty()) {
        // Texture asset paths need to be uniquely identifiable in order for Omniverse texture caching to function correctly.
        // If the tile url doesn't exist (which is unlikely), fall back to the tile name. Just be aware that the tile name
        // is not guaranteed to be the same across app invocations and caching may break.
        CESIUM_LOG_WARN("Could not get tile url. Texture caching may not function correctly.");
        texturePrefix = getTileNameUnique(tilesetId, tileId);
    }

    // Replace path separators and other special characters with underscores.
    // As of Kit 104.1, the texture cache only includes the name after the last path separator in the cache key.
    texturePrefix = UsdUtil::getSafeName(texturePrefix);

    const auto textureName = fmt::format("{}_texture_{}", texturePrefix, textureId);
    return getAssetPath(textureName, "bmp");
}

AssetPath getRasterOverlayAssetPath(const std::string& name, const CesiumGeometry::Rectangle& rectangle) {
    // Raster overlay paths need to be uniquely identifiable in order for Omniverse texture caching to function correctly.
    // Include both the name of the raster overlay and the region it covers. Since multiple raster overlay tiles may be
    // associated with a single geometry tile (e.g. Web Mercator imagery draped on WGS84 terrain) we don't have a single
    // url that we can use.
    const auto assetName = fmt::format(
        "{}_{}_{}_{}_{}", name, rectangle.minimumX, rectangle.minimumY, rectangle.maximumX, rectangle.maximumY);

    return getAssetPath(assetName, "bmp");
}

pxr::SdfPath getGeomPath(int64_t tilesetId, int64_t tileId, uint64_t primitiveId) {
    const auto tileName = getTileNameUnique(tilesetId, tileId);
    return pxr::SdfPath(fmt::format("/{}_primitive_{}", tileName, primitiveId));
}

pxr::SdfPath getMaterialPath(int64_t tilesetId, int64_t tileId, uint64_t materialId) {
    const auto tileName = getTileNameUnique(tilesetId, tileId);
    return pxr::SdfPath(fmt::format("/{}_material_{}", tileName, materialId));
}

std::vector<pxr::SdfPath> addMaterial(
    int64_t tilesetId,
    int64_t tileId,
    const pxr::SdfPath& materialPath,
    const std::vector<pxr::SdfAssetPath>& textureAssetPaths,
    const CesiumGltf::Model& model,
    const CesiumGltf::Material& material) {

    const auto baseColorTextureIndex = GltfUtil::getBaseColorTextureIndex(model, material);
    const auto baseColorFactor = GltfUtil::getBaseColorFactor(material);
    const auto hasDiffuseTexture = baseColorTextureIndex.has_value();

    auto sip = UsdUtil::getFabricStageInProgress();

    const auto shaderPath = materialPath.AppendChild(UsdTokens::Shader);
    const auto displacementPath = materialPath.AppendChild(UsdTokens::displacement);
    const auto surfacePath = materialPath.AppendChild(UsdTokens::surface);

    const auto shaderPathFabricUint64 = carb::flatcache::asInt(shaderPath).path;

    // Material
    {
        const auto materialPathFabric = carb::flatcache::Path(carb::flatcache::asInt(materialPath));
        sip.createPrim(materialPathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::_terminals, FabricTokens::_terminals);
        attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(materialPathFabric);

        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminals, 2);

        auto terminalsFabric = sip.getArrayAttributeWr<uint64_t>(materialPathFabric, FabricTokens::_terminals);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(materialPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(materialPathFabric, FabricTokens::_cesium_tileId);

        terminalsFabric[0] = shaderPathFabricUint64;
        terminalsFabric[1] = shaderPathFabricUint64;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // Displacement
    {
        const auto displacementPathFabric = carb::flatcache::Path(carb::flatcache::asInt(displacementPath));
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

        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_nodePaths, 1);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_inputId, 0);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_outputId, 0);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_inputName, 0);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_outputName, 0);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::primvars, 0);

        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_nodePaths);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(displacementPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(displacementPathFabric, FabricTokens::_cesium_tileId);

        nodePathsFabric[0] = shaderPathFabricUint64;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // Surface
    {
        const auto surfacePathFabric = carb::flatcache::Path(carb::flatcache::asInt(surfacePath));
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

        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_nodePaths, 1);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_inputId, 0);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_outputId, 0);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_inputName, 0);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_outputName, 0);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::primvars, 0);

        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_nodePaths);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(surfacePathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(surfacePathFabric, FabricTokens::_cesium_tileId);

        nodePathsFabric[0] = shaderPathFabricUint64;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // Shader
    {
        const auto shaderPathFabric = carb::flatcache::Path(carb::flatcache::asInt(shaderPath));
        sip.createPrim(shaderPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::diffuse_color_constant, FabricTokens::diffuse_color_constant);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        if (hasDiffuseTexture) {
            attributes.addAttribute(FabricTypes::diffuse_texture, FabricTokens::diffuse_texture);
        }

        attributes.createAttributes(shaderPathFabric);

        const size_t paramColorSpaceSize = hasDiffuseTexture ? 2 : 0;
        const size_t parametersCount = hasDiffuseTexture ? 2 : 1;
        const size_t parametersIndexDiffuseColorConstant = 0;
        const size_t parametersIndexDiffuseTexture = 1;

        sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::_paramColorSpace, paramColorSpaceSize);
        sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::_parameters, parametersCount);

        // clang-format off
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::_parameters);
        auto diffuseColorConstantFabric = sip.getAttributeWr<pxr::GfVec3f>(shaderPathFabric, FabricTokens::diffuse_color_constant);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(shaderPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(shaderPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        *infoIdFabric = FabricTokens::OmniPBR_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::OmniPBR;
        parametersFabric[parametersIndexDiffuseColorConstant] = FabricTokens::diffuse_color_constant;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;

        if (hasDiffuseTexture) {
            // clang-format off
            auto paramColorSpaceFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::_paramColorSpace);
            auto diffuseTextureFabric = sip.getAttributeWr<FabricAsset>(shaderPathFabric, FabricTokens::diffuse_texture);
            // clang-format on

            paramColorSpaceFabric[0] = FabricTokens::diffuse_texture;
            paramColorSpaceFabric[1] = FabricTokens::_auto;

            *diffuseColorConstantFabric = TEXTURE_LOADING_COLOR;
            parametersFabric[parametersIndexDiffuseTexture] = FabricTokens::diffuse_texture;
            *diffuseTextureFabric = FabricAsset(textureAssetPaths[baseColorTextureIndex.value()]);
        } else {
            *diffuseColorConstantFabric = baseColorFactor;
        }
    }

    return {
        materialPath,
        shaderPath,
        displacementPath,
        surfacePath,
    };
}

std::vector<pxr::SdfPath> addMaterialRasterOverlay(
    int64_t tilesetId,
    int64_t tileId,
    const pxr::SdfPath& materialPath,
    const pxr::SdfAssetPath& textureAssetPath,
    const glm::dvec2& uvTranslation,
    const glm::dvec2& uvScale) {
    auto sip = UsdUtil::getFabricStageInProgress();

    const auto shaderPath = materialPath.AppendChild(UsdTokens::Shader);
    const auto displacementPath = materialPath.AppendChild(UsdTokens::displacement);
    const auto surfacePath = materialPath.AppendChild(UsdTokens::surface);
    const auto multiplyPath = materialPath.AppendChild(UsdTokens::multiply);
    const auto addPath = materialPath.AppendChild(UsdTokens::add);
    const auto lookupColorPath = materialPath.AppendChild(UsdTokens::lookup_color);
    const auto textureCoordinate2dPath = materialPath.AppendChild(UsdTokens::texture_coordinate_2d);

    const auto shaderPathFabricUint64 = carb::flatcache::asInt(shaderPath).path;
    const auto multiplyPathFabricUint64 = carb::flatcache::asInt(multiplyPath).path;
    const auto addPathFabricUint64 = carb::flatcache::asInt(addPath).path;
    const auto lookupColorPathFabricUint64 = carb::flatcache::asInt(lookupColorPath).path;
    const auto textureCoordinate2dPathFabricUint64 = carb::flatcache::asInt(textureCoordinate2dPath).path;

    // Material
    {
        const auto materialPathFabric = carb::flatcache::Path(carb::flatcache::asInt(materialPath));
        sip.createPrim(materialPathFabric);

        FabricAttributesBuilder attributes;

        attributes.addAttribute(FabricTypes::_terminals, FabricTokens::_terminals);
        attributes.addAttribute(FabricTypes::Material, FabricTokens::Material);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);

        attributes.createAttributes(materialPathFabric);

        sip.setArrayAttributeSize(materialPathFabric, FabricTokens::_terminals, 2);

        auto terminalsFabric = sip.getArrayAttributeWr<uint64_t>(materialPathFabric, FabricTokens::_terminals);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(materialPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(materialPathFabric, FabricTokens::_cesium_tileId);

        terminalsFabric[0] = shaderPathFabricUint64;
        terminalsFabric[1] = shaderPathFabricUint64;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // Displacement
    {
        const auto displacementPathFabric = carb::flatcache::Path(carb::flatcache::asInt(displacementPath));
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

        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_nodePaths, 5);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_inputId, 4);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_outputId, 4);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_inputName, 4);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::_relationships_outputName, 4);
        sip.setArrayAttributeSize(displacementPathFabric, FabricTokens::primvars, 0);

        // clang-format off
        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_nodePaths);
        auto relationshipsInputIdFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_relationships_inputId);
        auto relationshipsOutputIdFabric = sip.getArrayAttributeWr<uint64_t>(displacementPathFabric, FabricTokens::_relationships_outputId);
        auto relationshipsInputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(displacementPathFabric, FabricTokens::_relationships_inputName);
        auto relationshipsOutputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(displacementPathFabric, FabricTokens::_relationships_outputName);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(displacementPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(displacementPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        nodePathsFabric[0] = textureCoordinate2dPathFabricUint64;
        nodePathsFabric[1] = multiplyPathFabricUint64;
        nodePathsFabric[2] = addPathFabricUint64;
        nodePathsFabric[3] = lookupColorPathFabricUint64;
        nodePathsFabric[4] = shaderPathFabricUint64;

        relationshipsInputIdFabric[0] = textureCoordinate2dPathFabricUint64;
        relationshipsInputIdFabric[1] = multiplyPathFabricUint64;
        relationshipsInputIdFabric[2] = addPathFabricUint64;
        relationshipsInputIdFabric[3] = lookupColorPathFabricUint64;

        relationshipsOutputIdFabric[0] = multiplyPathFabricUint64;
        relationshipsOutputIdFabric[1] = addPathFabricUint64;
        relationshipsOutputIdFabric[2] = lookupColorPathFabricUint64;
        relationshipsOutputIdFabric[3] = lookupColorPathFabricUint64;

        relationshipsInputNameFabric[0] = FabricTokens::out;
        relationshipsInputNameFabric[1] = FabricTokens::out;
        relationshipsInputNameFabric[2] = FabricTokens::out;
        relationshipsInputNameFabric[3] = FabricTokens::out;

        relationshipsOutputNameFabric[0] = FabricTokens::a;
        relationshipsOutputNameFabric[1] = FabricTokens::a;
        relationshipsOutputNameFabric[2] = FabricTokens::coord;
        relationshipsOutputNameFabric[3] = FabricTokens::diffuse_color_constant;

        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // Surface
    {
        const auto surfacePathFabric = carb::flatcache::Path(carb::flatcache::asInt(surfacePath));
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

        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_nodePaths, 5);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_inputId, 4);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_outputId, 4);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_inputName, 4);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::_relationships_outputName, 4);
        sip.setArrayAttributeSize(surfacePathFabric, FabricTokens::primvars, 0);

        // clang-format off
        auto nodePathsFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_nodePaths);
        auto relationshipsInputIdFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_relationships_inputId);
        auto relationshipsOutputIdFabric = sip.getArrayAttributeWr<uint64_t>(surfacePathFabric, FabricTokens::_relationships_outputId);
        auto relationshipsInputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(surfacePathFabric, FabricTokens::_relationships_inputName);
        auto relationshipsOutputNameFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(surfacePathFabric, FabricTokens::_relationships_outputName);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(surfacePathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(surfacePathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        nodePathsFabric[0] = textureCoordinate2dPathFabricUint64;
        nodePathsFabric[1] = multiplyPathFabricUint64;
        nodePathsFabric[2] = addPathFabricUint64;
        nodePathsFabric[3] = lookupColorPathFabricUint64;
        nodePathsFabric[4] = shaderPathFabricUint64;

        relationshipsInputIdFabric[0] = textureCoordinate2dPathFabricUint64;
        relationshipsInputIdFabric[1] = multiplyPathFabricUint64;
        relationshipsInputIdFabric[2] = addPathFabricUint64;
        relationshipsInputIdFabric[3] = lookupColorPathFabricUint64;

        relationshipsOutputIdFabric[0] = multiplyPathFabricUint64;
        relationshipsOutputIdFabric[1] = addPathFabricUint64;
        relationshipsOutputIdFabric[2] = lookupColorPathFabricUint64;
        relationshipsOutputIdFabric[3] = shaderPathFabricUint64;

        relationshipsInputNameFabric[0] = FabricTokens::out;
        relationshipsInputNameFabric[1] = FabricTokens::out;
        relationshipsInputNameFabric[2] = FabricTokens::out;
        relationshipsInputNameFabric[3] = FabricTokens::out;

        relationshipsOutputNameFabric[0] = FabricTokens::a;
        relationshipsOutputNameFabric[1] = FabricTokens::a;
        relationshipsOutputNameFabric[2] = FabricTokens::coord;
        relationshipsOutputNameFabric[3] = FabricTokens::diffuse_color_constant;

        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // Shader
    {
        const auto shaderPathFabric = carb::flatcache::Path(carb::flatcache::asInt(shaderPath));
        sip.createPrim(shaderPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::albedo_add, FabricTokens::albedo_add);
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        attributes.createAttributes(shaderPathFabric);

        sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::_paramColorSpace, 0);
        sip.setArrayAttributeSize(shaderPathFabric, FabricTokens::_parameters, 1);

        // clang-format off
        auto albedoAddFabric = sip.getAttributeWr<float>(shaderPathFabric, FabricTokens::albedo_add);
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(shaderPathFabric, FabricTokens::_parameters);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(shaderPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(shaderPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        *albedoAddFabric = 0.0f;
        *infoIdFabric = FabricTokens::OmniPBR_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::OmniPBR;
        parametersFabric[0] = FabricTokens::albedo_add;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // texture_coordinate_2d
    {
        const auto textureCoordinate2dPathFabric =
            carb::flatcache::Path(carb::flatcache::asInt(textureCoordinate2dPath));
        sip.createPrim(textureCoordinate2dPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        attributes.createAttributes(textureCoordinate2dPathFabric);

        sip.setArrayAttributeSize(textureCoordinate2dPathFabric, FabricTokens::_paramColorSpace, 0);
        sip.setArrayAttributeSize(textureCoordinate2dPathFabric, FabricTokens::_parameters, 0);

        // clang-format off
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(textureCoordinate2dPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(textureCoordinate2dPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(textureCoordinate2dPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(textureCoordinate2dPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        *infoIdFabric = FabricTokens::nvidia_support_definitions_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::texture_coordinate_2d;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // multiply
    {
        const auto multiplyPathFabric = carb::flatcache::Path(carb::flatcache::asInt(multiplyPath));
        sip.createPrim(multiplyPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::b, FabricTokens::b);
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        attributes.createAttributes(multiplyPathFabric);

        sip.setArrayAttributeSize(multiplyPathFabric, FabricTokens::_paramColorSpace, 0);
        sip.setArrayAttributeSize(multiplyPathFabric, FabricTokens::_parameters, 1);

        // clang-format off
        auto bFabric = sip.getAttributeWr<pxr::GfVec2f>(multiplyPathFabric, FabricTokens::b);
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(multiplyPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(multiplyPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(multiplyPathFabric, FabricTokens::_parameters);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(multiplyPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(multiplyPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        *bFabric = pxr::GfVec2f(static_cast<float>(uvScale.x), static_cast<float>(uvScale.y));
        *infoIdFabric = FabricTokens::nvidia_support_definitions_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::multiply_float2_float2;
        parametersFabric[0] = FabricTokens::b;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // add
    {
        const auto addPathFabric = carb::flatcache::Path(carb::flatcache::asInt(addPath));
        sip.createPrim(addPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::b, FabricTokens::b);
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        attributes.createAttributes(addPathFabric);

        sip.setArrayAttributeSize(addPathFabric, FabricTokens::_paramColorSpace, 0);
        sip.setArrayAttributeSize(addPathFabric, FabricTokens::_parameters, 1);

        // clang-format off
        auto bFabric = sip.getAttributeWr<pxr::GfVec2f>(addPathFabric, FabricTokens::b);
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(addPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(addPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(addPathFabric, FabricTokens::_parameters);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(addPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(addPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        *bFabric = pxr::GfVec2f(static_cast<float>(uvTranslation.x), static_cast<float>(uvTranslation.y));
        *infoIdFabric = FabricTokens::nvidia_support_definitions_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::add_float2_float2;
        parametersFabric[0] = FabricTokens::b;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    // lookup_color
    {
        const auto lookupColorPathFabric = carb::flatcache::Path(carb::flatcache::asInt(lookupColorPath));
        sip.createPrim(lookupColorPathFabric);

        FabricAttributesBuilder attributes;

        // clang-format off
        attributes.addAttribute(FabricTypes::wrap_u, FabricTokens::wrap_u);
        attributes.addAttribute(FabricTypes::wrap_v, FabricTokens::wrap_v);
        attributes.addAttribute(FabricTypes::tex, FabricTokens::tex);
        attributes.addAttribute(FabricTypes::info_id, FabricTokens::info_id);
        attributes.addAttribute(FabricTypes::info_sourceAsset_subIdentifier, FabricTokens::info_sourceAsset_subIdentifier);
        attributes.addAttribute(FabricTypes::_paramColorSpace, FabricTokens::_paramColorSpace);
        attributes.addAttribute(FabricTypes::_parameters, FabricTokens::_parameters);
        attributes.addAttribute(FabricTypes::Shader, FabricTokens::Shader);
        attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
        attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
        // clang-format on

        attributes.createAttributes(lookupColorPathFabric);

        sip.setArrayAttributeSize(lookupColorPathFabric, FabricTokens::_paramColorSpace, 2);
        sip.setArrayAttributeSize(lookupColorPathFabric, FabricTokens::_parameters, 3);

        // clang-format off
        auto wrapUFabric = sip.getAttributeWr<int>(lookupColorPathFabric, FabricTokens::wrap_u);
        auto wrapVFabric = sip.getAttributeWr<int>(lookupColorPathFabric, FabricTokens::wrap_v);
        auto texFabric = sip.getAttributeWr<FabricAsset>(lookupColorPathFabric, FabricTokens::tex);
        auto infoIdFabric = sip.getAttributeWr<carb::flatcache::Token>(lookupColorPathFabric, FabricTokens::info_id);
        auto infoSourceAssetSubIdentifierFabric = sip.getAttributeWr<carb::flatcache::Token>(lookupColorPathFabric, FabricTokens::info_sourceAsset_subIdentifier);
        auto paramColorSpaceFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(lookupColorPathFabric, FabricTokens::_paramColorSpace);
        auto parametersFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(lookupColorPathFabric, FabricTokens::_parameters);
        auto tilesetIdFabric = sip.getAttributeWr<int64_t>(lookupColorPathFabric, FabricTokens::_cesium_tilesetId);
        auto tileIdFabric = sip.getAttributeWr<int64_t>(lookupColorPathFabric, FabricTokens::_cesium_tileId);
        // clang-format on

        *wrapUFabric = 0;
        *wrapVFabric = 0;
        *texFabric = FabricAsset(textureAssetPath);
        *infoIdFabric = FabricTokens::nvidia_support_definitions_mdl;
        *infoSourceAssetSubIdentifierFabric = FabricTokens::lookup_color;
        paramColorSpaceFabric[0] = FabricTokens::tex;
        paramColorSpaceFabric[1] = FabricTokens::_auto;
        parametersFabric[0] = FabricTokens::tex;
        parametersFabric[1] = FabricTokens::wrap_u;
        parametersFabric[2] = FabricTokens::wrap_v;
        *tilesetIdFabric = tilesetId;
        *tileIdFabric = tileId;
    }

    return {
        materialPath,
        shaderPath,
        displacementPath,
        surfacePath,
        multiplyPath,
        addPath,
        lookupColorPath,
        textureCoordinate2dPath,
    };
}

void addPrimitive(
    int64_t tilesetId,
    int64_t tileId,
    const pxr::SdfPath& geomPath,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& gltfToEcefTransform,
    const glm::dmat4& nodeTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const std::vector<pxr::SdfPath>& materialPaths,
    uint64_t rasterOverlaySetIndex) {

    auto sip = UsdUtil::getFabricStageInProgress();
    const auto geomPathFabric = carb::flatcache::Path(carb::flatcache::asInt(geomPath));

    const auto positions = GltfUtil::getPrimitivePositions(model, primitive);
    const auto indices = GltfUtil::getPrimitiveIndices(model, primitive, positions);
    const auto normals = GltfUtil::getPrimitiveNormals(model, primitive, positions, indices);
    const auto st0 = GltfUtil::getPrimitiveUVs(model, primitive, 0);
    const auto rasterOverlaySt = GltfUtil::getRasterOverlayUVs(model, primitive, rasterOverlaySetIndex);
    const auto worldExtent = GltfUtil::getPrimitiveExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getPrimitiveFaceVertexCounts(indices);
    const auto doubleSided = GltfUtil::getDoubleSided(model, primitive);

    if (positions.empty() || indices.empty() || normals.empty() || !worldExtent.has_value()) {
        return;
    }

    int materialId = -1;

    if (!disableMaterials() && primitive.material >= 0 &&
        static_cast<size_t>(primitive.material) < materialPaths.size()) {
        materialId = primitive.material;
    }

    const auto hasMaterial = materialId != -1;
    const auto hasPrimitiveSt = !st0.empty();
    const auto hasRasterOverlaySt = !rasterOverlaySt.empty();
    const auto hasSt = hasPrimitiveSt || hasRasterOverlaySt;

    const auto localToEcefTransform = gltfToEcefTransform * nodeTransform;
    const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
    const auto [worldPosition, worldOrientation, worldScale] = UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);

    sip.createPrim(geomPathFabric);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars, FabricTokens::primvars);
    attributes.addAttribute(FabricTypes::primvarInterpolations, FabricTokens::primvarInterpolations);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::primvars_normals, FabricTokens::primvars_normals);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
    attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
    attributes.addAttribute(FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided);
    attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme);

    if (hasMaterial) {
        attributes.addAttribute(FabricTypes::materialId, FabricTokens::materialId);
    }

    if (hasSt) {
        attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st);
    }

    attributes.createAttributes(geomPathFabric);

    const size_t primvarsCount = hasSt ? 3 : 2;
    const size_t primvarIndexDisplayColor = 0;
    const size_t primvarIndexNormal = 1;
    const size_t primvarIndexSt = 2;

    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::faceVertexCounts, faceVertexCounts.size());
    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::faceVertexIndices, indices.size());
    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::points, positions.size());
    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::primvars, primvarsCount);
    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::primvarInterpolations, primvarsCount);
    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::primvars_displayColor, 1);
    sip.setArrayAttributeSize(geomPathFabric, FabricTokens::primvars_normals, normals.size());

    // clang-format off
    auto faceVertexCountsFabric = sip.getArrayAttributeWr<int>(geomPathFabric, FabricTokens::faceVertexCounts);
    auto faceVertexIndicesFabric = sip.getArrayAttributeWr<int>(geomPathFabric, FabricTokens::faceVertexIndices);
    auto pointsFabric = sip.getArrayAttributeWr<pxr::GfVec3f>(geomPathFabric, FabricTokens::points);
    auto worldExtentFabric = sip.getAttributeWr<pxr::GfRange3d>(geomPathFabric, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = sip.getAttributeWr<bool>(geomPathFabric, FabricTokens::_worldVisibility);
    auto primvarsFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(geomPathFabric, FabricTokens::primvars);
    auto primvarInterpolationsFabric = sip.getArrayAttributeWr<carb::flatcache::Token>(geomPathFabric, FabricTokens::primvarInterpolations);
    auto displayColorFabric = sip.getArrayAttributeWr<pxr::GfVec3f>(geomPathFabric, FabricTokens::primvars_displayColor);
    auto normalsFabric = sip.getArrayAttributeWr<pxr::GfVec3f>(geomPathFabric, FabricTokens::primvars_normals);
    auto tilesetIdFabric = sip.getAttributeWr<int64_t>(geomPathFabric, FabricTokens::_cesium_tilesetId);
    auto tileIdFabric = sip.getAttributeWr<int64_t>(geomPathFabric, FabricTokens::_cesium_tileId);
    auto localToEcefTransformFabric = sip.getAttributeWr<pxr::GfMatrix4d>(geomPathFabric, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = sip.getAttributeWr<pxr::GfVec3d>(geomPathFabric, FabricTokens::_worldPosition);
    auto worldOrientationFabric = sip.getAttributeWr<pxr::GfQuatf>(geomPathFabric, FabricTokens::_worldOrientation);
    auto worldScaleFabric = sip.getAttributeWr<pxr::GfVec3f>(geomPathFabric, FabricTokens::_worldScale);
    auto doubleSidedFabric = sip.getAttributeWr<bool>(geomPathFabric, FabricTokens::doubleSided);
    auto subdivisionSchemeFabric = sip.getAttributeWr<carb::flatcache::Token>(geomPathFabric, FabricTokens::subdivisionScheme);
    // clang-format on

    std::copy(faceVertexCounts.begin(), faceVertexCounts.end(), faceVertexCountsFabric.begin());
    std::copy(indices.begin(), indices.end(), faceVertexIndicesFabric.begin());
    std::copy(positions.begin(), positions.end(), pointsFabric.begin());
    std::copy(normals.begin(), normals.end(), normalsFabric.begin());

    *worldVisibilityFabric = false;
    primvarsFabric[primvarIndexDisplayColor] = FabricTokens::primvars_displayColor;
    primvarInterpolationsFabric[primvarIndexDisplayColor] = FabricTokens::constant;
    primvarsFabric[primvarIndexNormal] = FabricTokens::primvars_normals;
    primvarInterpolationsFabric[primvarIndexNormal] = FabricTokens::vertex;
    *tilesetIdFabric = tilesetId;
    *tileIdFabric = tileId;
    *worldPositionFabric = worldPosition;
    *worldOrientationFabric = worldOrientation;
    *worldScaleFabric = worldScale;
    *localToEcefTransformFabric = UsdUtil::glmToUsdMatrix(localToEcefTransform);
    *doubleSidedFabric = doubleSided;
    *subdivisionSchemeFabric = FabricTokens::none;

    worldExtentFabric->SetMin(worldExtent.value().GetMin());
    worldExtentFabric->SetMax(worldExtent.value().GetMax());

    if (hasMaterial) {
        auto materialIdFabric = sip.getAttributeWr<uint64_t>(geomPathFabric, FabricTokens::materialId);
        *materialIdFabric = carb::flatcache::asInt(materialPaths[static_cast<size_t>(materialId)]).path;
        displayColorFabric[0] = MATERIAL_LOADING_COLOR;
    } else {
        displayColorFabric[0] = DEFAULT_COLOR;
    }

    if (hasSt) {
        const auto& st = hasRasterOverlaySt ? rasterOverlaySt : st0;

        sip.setArrayAttributeSize(geomPathFabric, FabricTokens::primvars_st, st.size());

        auto stFabric = sip.getArrayAttributeWr<pxr::GfVec2f>(geomPathFabric, FabricTokens::primvars_st);

        std::copy(st.begin(), st.end(), stFabric.begin());

        primvarsFabric[primvarIndexSt] = FabricTokens::primvars_st;
        primvarInterpolationsFabric[primvarIndexSt] = FabricTokens::vertex;
    }
}

std::vector<std::byte> convertImageToBmp(const CesiumGltf::ImageCesium& image) {
    std::vector<std::byte> writeData;
    stbi_write_bmp_to_func(
        [](void* context, void* data, int size) {
            auto& write = *reinterpret_cast<std::vector<std::byte>*>(context);
            const auto bdata = reinterpret_cast<std::byte*>(data);
            write.insert(write.end(), bdata, bdata + size);
        },
        &writeData,
        image.width,
        image.height,
        image.channels,
        image.pixelData.data());

    return writeData;
}

void addTexture(const std::string& assetName, const CesiumGltf::ImageCesium& image) {
    auto inMemoryAsset = std::make_shared<pxr::InMemoryAsset>(convertImageToBmp(image));
    auto& ctx = pxr::InMemoryAssetContext::instance();
    ctx.assets.insert({assetName, std::move(inMemoryAsset)});
}

void removeTexture(const std::string& assetName) {
    auto& ctx = pxr::InMemoryAssetContext::instance();
    ctx.assets.erase(assetName);
}

void deletePrimsFabric(const std::vector<pxr::SdfPath>& primsToDelete) {
    // Prims removed from Fabric need special handling for their removal to be reflected in the Hydra render index
    // This workaround may not be needed in future Kit versions, but is needed as of Kit 104.2
    auto sip = UsdUtil::getFabricStageInProgress();

    const carb::flatcache::Path changeTrackingPath("/TempChangeTracking");

    if (sip.getAttribute<uint64_t>(changeTrackingPath, FabricTokens::_deletedPrims) == nullptr) {
        return;
    }

    const auto deletedPrimsSize = sip.getArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims);
    sip.setArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims, deletedPrimsSize + primsToDelete.size());
    auto deletedPrimsFabric = sip.getArrayAttributeWr<uint64_t>(changeTrackingPath, FabricTokens::_deletedPrims);

    for (size_t i = 0; i < primsToDelete.size(); i++) {
        deletedPrimsFabric[deletedPrimsSize + i] = carb::flatcache::asInt(primsToDelete[i]).path;
    }
}

void deletePrimsFabric(const std::vector<uint64_t>& primsToDelete) {
    // Prims removed from Fabric need special handling for their removal to be reflected in the Hydra render index
    // This workaround may not be needed in future Kit versions, but is needed as of Kit 104.2
    auto sip = UsdUtil::getFabricStageInProgress();

    const carb::flatcache::Path changeTrackingPath("/TempChangeTracking");

    if (sip.getAttribute<uint64_t>(changeTrackingPath, FabricTokens::_deletedPrims) == nullptr) {
        return;
    }

    const auto deletedPrimsSize = sip.getArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims);
    sip.setArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims, deletedPrimsSize + primsToDelete.size());
    auto deletedPrimsFabric = sip.getArrayAttributeWr<uint64_t>(changeTrackingPath, FabricTokens::_deletedPrims);

    for (size_t i = 0; i < primsToDelete.size(); i++) {
        deletedPrimsFabric[deletedPrimsSize + i] = primsToDelete[i];
    }
}

} // namespace

AddTileResults addTile(
    int64_t tilesetId,
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& tileTransform,
    const CesiumGltf::Model& model) {
    auto gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyRtcCenter(model, tileTransform);
    gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyGltfUpAxisTransform(model, gltfToEcefTransform);

    std::vector<std::string> textureAssetNames;
    std::vector<pxr::SdfAssetPath> textureAssetPaths;
    std::vector<pxr::SdfPath> materialPaths;
    std::vector<pxr::SdfPath> allPrimPaths;

    if (!disableMaterials()) {
        textureAssetPaths.reserve(model.textures.size());
        textureAssetNames.reserve(model.textures.size());
        for (size_t i = 0; i < model.textures.size(); ++i) {
            const auto textureAssetPath = getTextureAssetPath(model, tilesetId, tileId, i);
            textureAssetPaths.push_back(textureAssetPath.assetPath);
            textureAssetNames.push_back(textureAssetPath.assetName);
            addTexture(textureAssetPath.assetName, GltfUtil::getImageCesium(model, model.textures[i]));
        }

        materialPaths.reserve(model.materials.size());

        for (size_t i = 0; i < model.materials.size(); i++) {
            auto materialPath = getMaterialPath(tilesetId, tileId, i);
            const auto materialPrimPaths =
                addMaterial(tilesetId, tileId, materialPath, textureAssetPaths, model, model.materials[i]);

            materialPaths.emplace_back(std::move(materialPath));
            allPrimPaths.insert(
                allPrimPaths.begin(),
                std::make_move_iterator(materialPrimPaths.begin()),
                std::make_move_iterator(materialPrimPaths.end()));
        }
    }

    uint64_t primitiveId = 0;

    std::vector<pxr::SdfPath> geomPaths;

    model.forEachPrimitiveInScene(
        -1,
        [tilesetId, tileId, &primitiveId, &ecefToUsdTransform, &gltfToEcefTransform, &materialPaths, &geomPaths](
            const CesiumGltf::Model& gltf,
            [[maybe_unused]] const CesiumGltf::Node& node,
            [[maybe_unused]] const CesiumGltf::Mesh& mesh,
            const CesiumGltf::MeshPrimitive& primitive,
            const glm::dmat4& transform) {
            auto geomPath = getGeomPath(tilesetId, tileId, primitiveId++);
            addPrimitive(
                tilesetId,
                tileId,
                geomPath,
                ecefToUsdTransform,
                gltfToEcefTransform,
                transform,
                gltf,
                primitive,
                materialPaths,
                0);
            geomPaths.emplace_back(std::move(geomPath));
        });

    allPrimPaths.insert(allPrimPaths.begin(), geomPaths.begin(), geomPaths.end());

    return AddTileResults{geomPaths, allPrimPaths, textureAssetNames};
}

AddTileResults addTileWithRasterOverlay(
    int64_t tilesetId,
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& tileTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::ImageCesium& rasterOverlayImage,
    const std::string& rasterOverlayName,
    const CesiumGeometry::Rectangle& rasterOverlayRectangle,
    const glm::dvec2& rasterOverlayUvTranslation,
    const glm::dvec2& rasterOverlayUvScale,
    uint64_t rasterOverlayUvSetIndex) {
    auto gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyRtcCenter(model, tileTransform);
    gltfToEcefTransform = Cesium3DTilesSelection::GltfUtilities::applyGltfUpAxisTransform(model, gltfToEcefTransform);

    std::vector<std::string> textureAssetNames;
    std::vector<pxr::SdfAssetPath> textureAssetPaths;
    std::vector<pxr::SdfPath> materialPaths;
    std::vector<pxr::SdfPath> allPrimPaths;

    if (!disableMaterials()) {
        const auto rasterOverlayAssetPath = getRasterOverlayAssetPath(rasterOverlayName, rasterOverlayRectangle);
        addTexture(rasterOverlayAssetPath.assetName, rasterOverlayImage);
        textureAssetPaths.push_back(rasterOverlayAssetPath.assetPath);
        textureAssetNames.push_back(rasterOverlayAssetPath.assetName);

        materialPaths.reserve(model.materials.size());

        for (size_t i = 0; i < model.materials.size(); i++) {
            auto materialPath = getMaterialPath(tilesetId, tileId, i);
            const auto materialPrimPaths = addMaterialRasterOverlay(
                tilesetId,
                tileId,
                materialPath,
                rasterOverlayAssetPath.assetPath,
                rasterOverlayUvTranslation,
                rasterOverlayUvScale);

            materialPaths.emplace_back(std::move(materialPath));
            allPrimPaths.insert(
                allPrimPaths.begin(),
                std::make_move_iterator(materialPrimPaths.begin()),
                std::make_move_iterator(materialPrimPaths.end()));
        }
    }

    uint64_t primitiveId = 0;

    std::vector<pxr::SdfPath> geomPaths;

    model.forEachPrimitiveInScene(
        -1,
        [tilesetId,
         tileId,
         &primitiveId,
         &ecefToUsdTransform,
         &gltfToEcefTransform,
         &materialPaths,
         &geomPaths,
         rasterOverlayUvSetIndex](
            const CesiumGltf::Model& gltf,
            [[maybe_unused]] const CesiumGltf::Node& node,
            [[maybe_unused]] const CesiumGltf::Mesh& mesh,
            const CesiumGltf::MeshPrimitive& primitive,
            const glm::dmat4& transform) {
            auto geomPath = getGeomPath(tilesetId, tileId, primitiveId++);
            addPrimitive(
                tilesetId,
                tileId,
                geomPath,
                ecefToUsdTransform,
                gltfToEcefTransform,
                transform,
                gltf,
                primitive,
                materialPaths,
                rasterOverlayUvSetIndex);
            geomPaths.emplace_back(std::move(geomPath));
        });

    allPrimPaths.insert(allPrimPaths.begin(), geomPaths.begin(), geomPaths.end());

    return AddTileResults{geomPaths, allPrimPaths, textureAssetNames};
}

void removeTile(const std::vector<pxr::SdfPath>& allPrimPaths, const std::vector<std::string>& textureAssetNames) {
    auto sip = UsdUtil::getFabricStageInProgress();

    for (const auto& primPath : allPrimPaths) {
        sip.destroyPrim(carb::flatcache::asInt(primPath));
    }

    deletePrimsFabric(allPrimPaths);

    for (const auto& textureAssetName : textureAssetNames) {
        removeTexture(textureAssetName);
    }
}

void setTileVisibility(const std::vector<pxr::SdfPath>& geomPaths, bool visible) {
    auto sip = UsdUtil::getFabricStageInProgress();

    for (const auto& geomPath : geomPaths) {
        auto worldVisibilityFabric =
            sip.getAttributeWr<bool>(carb::flatcache::asInt(geomPath), FabricTokens::_worldVisibility);
        *worldVisibilityFabric = visible;
    }
}

void removeTileset(int64_t tilesetId) {
    auto sip = UsdUtil::getFabricStageInProgress();

    const auto buckets = sip.findPrims(
        {carb::flatcache::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)});

    std::vector<uint64_t> primsToDelete;

    for (size_t bucketId = 0; bucketId < buckets.bucketCount(); bucketId++) {
        const auto tilesetIdFabric =
            sip.getAttributeArrayRd<int64_t>(buckets, bucketId, FabricTokens::_cesium_tilesetId);
        const auto primPaths = sip.getPathArray(buckets, bucketId);

        for (size_t i = 0; i < tilesetIdFabric.size(); i++) {
            if (tilesetIdFabric[i] == tilesetId) {
                primsToDelete.push_back(carb::flatcache::PathC(primPaths[i]).path);
            }
        }
    }

    for (const auto& primToDelete : primsToDelete) {
        sip.destroyPrim(carb::flatcache::Path(carb::flatcache::PathC{primToDelete}));
    }

    deletePrimsFabric(primsToDelete);
}

void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform) {
    auto sip = UsdUtil::getFabricStageInProgress();

    const auto buckets = sip.findPrims(
        {carb::flatcache::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {carb::flatcache::AttrNameAndType(
            FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform)});

    for (size_t bucketId = 0; bucketId < buckets.bucketCount(); bucketId++) {
        // clang-format off
        auto tilesetIdFabric = sip.getAttributeArrayRd<int64_t>(buckets, bucketId, FabricTokens::_cesium_tilesetId);
        auto localToEcefTransformFabric = sip.getAttributeArrayRd<pxr::GfMatrix4d>(buckets, bucketId, FabricTokens::_cesium_localToEcefTransform);
        auto worldPositionFabric = sip.getAttributeArrayWr<pxr::GfVec3d>(buckets, bucketId, FabricTokens::_worldPosition);
        auto worldOrientationFabric = sip.getAttributeArrayWr<pxr::GfQuatf>(buckets, bucketId, FabricTokens::_worldOrientation);
        auto worldScaleFabric = sip.getAttributeArrayWr<pxr::GfVec3f>(buckets, bucketId, FabricTokens::_worldScale);
        // clang-format on

        for (size_t i = 0; i < tilesetIdFabric.size(); i++) {
            if (tilesetIdFabric[i] == tilesetId) {
                const auto localToEcefTransform = UsdUtil::usdToGlmMatrix(localToEcefTransformFabric[i]);
                const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
                const auto [position, orientation, scale] = UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);
                worldPositionFabric[i] = position;
                worldOrientationFabric[i] = orientation;
                worldScaleFabric[i] = scale;
            }
        }
    }
}

} // namespace cesium::omniverse::FabricStageUtil
