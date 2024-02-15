#include "cesium/omniverse/UsdNotificationHandler.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/OmniCartographicPolygon.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/OmniIonRasterOverlay.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/OmniPolygonRasterOverlay.h"
#include "cesium/omniverse/OmniRasterOverlay.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/OmniWebMapServiceRasterOverlay.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/tokens.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdShade/shader.h>

namespace cesium::omniverse {

namespace {

bool isPrimOrDescendant(const pxr::SdfPath& descendantPath, const pxr::SdfPath& path) {
    if (descendantPath == path) {
        return true;
    }

    for (const auto& ancestorPath : descendantPath.GetAncestorsRange()) {
        if (ancestorPath == path) {
            return true;
        }
    }

    return false;
}

void updateRasterOverlayBindings(const Context& context, const pxr::SdfPath& rasterOverlayPath) {
    const auto& tilesets = context.getAssetRegistry().getTilesets();

    // Update tilesets that reference this raster overlay
    for (const auto& pTileset : tilesets) {
        if (CppUtil::contains(pTileset->getRasterOverlayPaths(), rasterOverlayPath)) {
            pTileset->reload();
        }
    }
}

void updateRasterOverlayBindingsAlpha(const Context& context, const pxr::SdfPath& rasterOverlayPath) {
    const auto& tilesets = context.getAssetRegistry().getTilesets();

    // Update tilesets that reference this raster overlay
    for (const auto& pTileset : tilesets) {
        if (CppUtil::contains(pTileset->getRasterOverlayPaths(), rasterOverlayPath)) {
            pTileset->updateRasterOverlayAlpha(rasterOverlayPath);
        }
    }
}

void updateIonServerBindings(const Context& context) {
    // Update all tilesets. Some tilesets may have referenced this ion server implicitly.
    const auto& tilesets = context.getAssetRegistry().getTilesets();
    for (const auto& pTileset : tilesets) {
        pTileset->reload();
    }

    // Update all raster overlays. Some raster overlays may have referenced this ion server implicitly.
    const auto& ionRasterOverlays = context.getAssetRegistry().getIonRasterOverlays();
    for (const auto& pIonRasterOverlay : ionRasterOverlays) {
        pIonRasterOverlay->reload();
        updateRasterOverlayBindings(context, pIonRasterOverlay->getPath());
    }
}

void updateCartographicPolygonBindings(const Context& context, const pxr::SdfPath& cartographicPolygonPath) {
    // Update polygon raster overlays that reference this cartographic polygon
    const auto& polygonRasterOverlays = context.getAssetRegistry().getPolygonRasterOverlays();
    for (const auto& pPolygonRasterOverlay : polygonRasterOverlays) {
        const auto paths = pPolygonRasterOverlay->getCartographicPolygonPaths();
        if (CppUtil::contains(paths, cartographicPolygonPath)) {
            pPolygonRasterOverlay->reload();
            updateRasterOverlayBindings(context, pPolygonRasterOverlay->getPath());
        }
    }
}

void updateGlobeAnchorBindings(const Context& context, const pxr::SdfPath& globeAnchorPath) {
    // Don't need to update tilesets. Globe anchor changes are handled automatically in the update loop.

    if (context.getAssetRegistry().getCartographicPolygon(globeAnchorPath)) {
        // Update cartographic polygon that this globe anchor is attached to
        updateCartographicPolygonBindings(context, globeAnchorPath);
    }
}

void updateGeoreferenceBindings(const Context& context) {
    // Don't need to update tilesets. Georeference changes are handled automatically in the update loop.

    // Update all globe anchors. Some globe anchors may have referenced this georeference implicitly.
    const auto& globeAnchors = context.getAssetRegistry().getGlobeAnchors();
    for (const auto& pGlobeAnchor : globeAnchors) {
        pGlobeAnchor->updateByGeoreference();
        updateGlobeAnchorBindings(context, pGlobeAnchor->getPath());
    }
}

bool isFirstData(const Context& context, const pxr::SdfPath& dataPath) {
    const auto pData = context.getAssetRegistry().getData(dataPath);
    const auto pFirstData = context.getAssetRegistry().getFirstData();

    return pData && pData == pFirstData;
}

[[nodiscard]] bool processCesiumDataChanged(
    const Context& context,
    const pxr::SdfPath& dataPath,
    const std::vector<pxr::TfToken>& properties) {

    if (!isFirstData(context, dataPath)) {
        return false;
    }

    auto reloadStage = false;
    auto updateGeoreference = false;

    // No change tracking needed for
    // * selectedIonServer
    // * projectDefaultIonAccessToken (deprecated)
    // * projectDefaultIonAccessTokenId (deprecated)

    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumDebugDisableMaterials ||
            property == pxr::CesiumTokens->cesiumDebugDisableTextures ||
            property == pxr::CesiumTokens->cesiumDebugDisableGeometryPool ||
            property == pxr::CesiumTokens->cesiumDebugDisableMaterialPool ||
            property == pxr::CesiumTokens->cesiumDebugDisableTexturePool ||
            property == pxr::CesiumTokens->cesiumDebugGeometryPoolInitialCapacity ||
            property == pxr::CesiumTokens->cesiumDebugMaterialPoolInitialCapacity ||
            property == pxr::CesiumTokens->cesiumDebugTexturePoolInitialCapacity ||
            property == pxr::CesiumTokens->cesiumDebugRandomColors) {
            reloadStage = true;
        } else if (property == pxr::CesiumTokens->cesiumDebugDisableGeoreferencing) {
            updateGeoreference = true;
        }
    }

    if (updateGeoreference) {
        updateGeoreferenceBindings(context);
    }

    return reloadStage;
}

void processCesiumGlobeAnchorChanged(
    const Context& context,
    const pxr::SdfPath& globeAnchorPath,
    const std::vector<pxr::TfToken>& properties) {
    const auto pGlobeAnchor = context.getAssetRegistry().getGlobeAnchor(globeAnchorPath);
    if (!pGlobeAnchor) {
        return;
    }

    // No change tracking needed for
    // * adjustOrientation

    auto updateByGeoreference = false;
    auto updateByPrimLocalTransform = false;
    auto updateByGeographicCoordinates = false;
    auto updateByEcefPosition = false;
    auto updateBindings = false;
    auto resetOrientation = false;

    const auto detectTransformChanges = pGlobeAnchor->getDetectTransformChanges();

    // clang-format off
    for (const auto& property : properties) {
        if (detectTransformChanges &&
            (property == pxr::UsdTokens->xformOp_translate ||
             property == pxr::UsdTokens->xformOp_rotateXYZ ||
             property == pxr::UsdTokens->xformOp_rotateXZY ||
             property == pxr::UsdTokens->xformOp_rotateYXZ ||
             property == pxr::UsdTokens->xformOp_rotateYZX ||
             property == pxr::UsdTokens->xformOp_rotateZXY ||
             property == pxr::UsdTokens->xformOp_rotateZYX ||
             property == pxr::UsdTokens->xformOp_scale)) {
            updateByPrimLocalTransform = true;
            updateBindings = true;
        } else if (property == pxr::CesiumTokens->cesiumAnchorLongitude ||
            property == pxr::CesiumTokens->cesiumAnchorLatitude ||
            property == pxr::CesiumTokens->cesiumAnchorHeight) {
            updateByGeographicCoordinates = true;
            updateBindings = true;
        } else if (property == pxr::CesiumTokens->cesiumAnchorPosition) {
            updateByEcefPosition = true;
            updateBindings = true;
        } else if (property == pxr::CesiumTokens->cesiumAnchorGeoreferenceBinding) {
            updateByGeoreference = true;
            updateBindings = true;
        } else if (detectTransformChanges && property == pxr::CesiumTokens->cesiumAnchorDetectTransformChanges) {
            updateByPrimLocalTransform = true;
            updateBindings = true;
            resetOrientation = true;
        }
    }
    // clang-format on

    if (updateByGeoreference) {
        pGlobeAnchor->updateByGeoreference();
    }

    if (updateByEcefPosition) {
        pGlobeAnchor->updateByEcefPosition();
    }

    if (updateByGeographicCoordinates) {
        pGlobeAnchor->updateByGeographicCoordinates();
    }

    if (updateByPrimLocalTransform) {
        pGlobeAnchor->updateByPrimLocalTransform(resetOrientation);
    }

    if (updateBindings) {
        updateGlobeAnchorBindings(context, globeAnchorPath);
    }
}

void processCesiumTilesetChanged(
    const Context& context,
    const pxr::SdfPath& tilesetPath,
    const std::vector<pxr::TfToken>& properties) {
    const auto pTileset = context.getAssetRegistry().getTileset(tilesetPath);
    if (!pTileset) {
        return;
    }

    // Process globe anchor API schema first
    processCesiumGlobeAnchorChanged(context, tilesetPath, properties);

    auto reload = false;
    auto updateTilesetOptions = false;
    auto updateDisplayColorAndOpacity = false;

    // No change tracking needed for
    // * suspendUpdate
    // * georeferenceBinding
    // * Transform changes (handled automatically in update loop)

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumSourceType ||
            property == pxr::CesiumTokens->cesiumUrl ||
            property == pxr::CesiumTokens->cesiumIonAssetId ||
            property == pxr::CesiumTokens->cesiumIonAccessToken ||
            property == pxr::CesiumTokens->cesiumIonServerBinding ||
            property == pxr::CesiumTokens->cesiumSmoothNormals ||
            property == pxr::CesiumTokens->cesiumShowCreditsOnScreen ||
            property == pxr::CesiumTokens->cesiumRasterOverlayBinding ||
            property == pxr::UsdTokens->material_binding) {
            reload = true;
        } else if (
            property == pxr::CesiumTokens->cesiumMaximumScreenSpaceError ||
            property == pxr::CesiumTokens->cesiumPreloadAncestors ||
            property == pxr::CesiumTokens->cesiumPreloadSiblings ||
            property == pxr::CesiumTokens->cesiumForbidHoles ||
            property == pxr::CesiumTokens->cesiumMaximumSimultaneousTileLoads ||
            property == pxr::CesiumTokens->cesiumMaximumCachedBytes ||
            property == pxr::CesiumTokens->cesiumLoadingDescendantLimit ||
            property == pxr::CesiumTokens->cesiumEnableFrustumCulling ||
            property == pxr::CesiumTokens->cesiumEnableFogCulling ||
            property == pxr::CesiumTokens->cesiumEnforceCulledScreenSpaceError ||
            property == pxr::CesiumTokens->cesiumCulledScreenSpaceError ||
            property == pxr::CesiumTokens->cesiumMainThreadLoadingTimeLimit) {
            updateTilesetOptions = true;
        } else if (
            property == pxr::UsdTokens->primvars_displayColor ||
            property == pxr::UsdTokens->primvars_displayOpacity) {
            updateDisplayColorAndOpacity = true;
        }
    }
    // clang-format on

    if (reload) {
        pTileset->reload();
    }

    if (updateTilesetOptions) {
        pTileset->updateTilesetOptions();
    }

    if (updateDisplayColorAndOpacity) {
        pTileset->updateDisplayColorAndOpacity();
    }
}

void processCesiumRasterOverlayChanged(
    const Context& context,
    const pxr::SdfPath& rasterOverlayPath,
    const std::vector<pxr::TfToken>& properties) {
    const auto pRasterOverlay = context.getAssetRegistry().getRasterOverlay(rasterOverlayPath);
    if (!pRasterOverlay) {
        return;
    }

    auto reload = false;
    auto updateBindings = false;
    auto updateRasterOverlayAlpha = false;
    auto updateRasterOverlayOptions = false;

    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumShowCreditsOnScreen) {
            reload = true;
            updateBindings = true;
        } else if (property == pxr::CesiumTokens->cesiumOverlayRenderMethod) {
            updateBindings = true;
        } else if (property == pxr::CesiumTokens->cesiumAlpha) {
            updateRasterOverlayAlpha = true;
        } else if (
            property == pxr::CesiumTokens->cesiumMaximumScreenSpaceError ||
            property == pxr::CesiumTokens->cesiumMaximumTextureSize ||
            property == pxr::CesiumTokens->cesiumMaximumSimultaneousTileLoads ||
            property == pxr::CesiumTokens->cesiumSubTileCacheBytes) {
            updateRasterOverlayOptions = true;
        }
    }

    if (reload) {
        pRasterOverlay->reload();
    }

    if (updateBindings) {
        updateRasterOverlayBindings(context, rasterOverlayPath);
    }

    if (updateRasterOverlayAlpha) {
        updateRasterOverlayBindingsAlpha(context, rasterOverlayPath);
    }

    if (updateRasterOverlayOptions) {
        pRasterOverlay->updateRasterOverlayOptions();
    }
}

void processCesiumIonRasterOverlayChanged(
    const Context& context,
    const pxr::SdfPath& ionRasterOverlayPath,
    const std::vector<pxr::TfToken>& properties) {
    const auto pIonRasterOverlay = context.getAssetRegistry().getIonRasterOverlay(ionRasterOverlayPath);
    if (!pIonRasterOverlay) {
        return;
    }

    // Process base class first
    processCesiumRasterOverlayChanged(context, ionRasterOverlayPath, properties);

    auto reload = false;
    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumIonAssetId || property == pxr::CesiumTokens->cesiumIonAccessToken ||
            property == pxr::CesiumTokens->cesiumIonServerBinding) {
            reload = true;
            updateBindings = true;
        }
    }

    if (reload) {
        pIonRasterOverlay->reload();
    }

    if (updateBindings) {
        updateRasterOverlayBindings(context, ionRasterOverlayPath);
    }
}

void processCesiumPolygonRasterOverlayChanged(
    const Context& context,
    const pxr::SdfPath& polygonRasterOverlayPath,
    const std::vector<pxr::TfToken>& properties) {
    const auto pPolygonRasterOverlay = context.getAssetRegistry().getPolygonRasterOverlay(polygonRasterOverlayPath);
    if (!pPolygonRasterOverlay) {
        return;
    }

    // Process base class first
    processCesiumRasterOverlayChanged(context, polygonRasterOverlayPath, properties);

    auto reload = false;
    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumCartographicPolygonBinding ||
            property == pxr::CesiumTokens->cesiumInvertSelection) {
            reload = true;
            updateBindings = true;
        }
    }

    if (reload) {
        pPolygonRasterOverlay->reload();
    }

    if (updateBindings) {
        updateRasterOverlayBindings(context, polygonRasterOverlayPath);
    }
}

void processCesiumWebMapServiceRasterOverlayChanged(
    const Context& context,
    const pxr::SdfPath& webMapServiceRasterOverlayPath,
    const std::vector<pxr::TfToken>& properties) {

    const auto pWebMapServiceRasterOverlay =
        context.getAssetRegistry().getWebMapServiceRasterOverlay(webMapServiceRasterOverlayPath);
    if (!pWebMapServiceRasterOverlay) {
        return;
    }

    // Process base class first
    processCesiumRasterOverlayChanged(context, webMapServiceRasterOverlayPath, properties);

    auto reload = false;
    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumBaseUrl || property == pxr::CesiumTokens->cesiumLayers ||
            property == pxr::CesiumTokens->cesiumTileWidth || property == pxr::CesiumTokens->cesiumTileHeight ||
            property == pxr::CesiumTokens->cesiumMinimumLevel || property == pxr::CesiumTokens->cesiumMaximumLevel) {
            reload = true;
            updateBindings = true;
        }
    }

    if (reload) {
        pWebMapServiceRasterOverlay->reload();
    }

    if (updateBindings) {
        updateRasterOverlayBindings(context, webMapServiceRasterOverlayPath);
    }
}

void processCesiumGeoreferenceChanged(const Context& context, const std::vector<pxr::TfToken>& properties) {

    auto updateBindings = false;

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumGeoreferenceOriginLongitude ||
            property == pxr::CesiumTokens->cesiumGeoreferenceOriginLatitude ||
            property == pxr::CesiumTokens->cesiumGeoreferenceOriginHeight) {
            updateBindings = true;
        }
    }
    // clang-format on

    if (updateBindings) {
        updateGeoreferenceBindings(context);
    }
}

void processCesiumIonServerChanged(
    Context& context,
    const pxr::SdfPath& ionServerPath,
    const std::vector<pxr::TfToken>& properties) {

    auto reloadSession = false;
    auto updateBindings = false;

    // No change tracking needed for
    // * displayName

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumIonServerUrl ||
            property == pxr::CesiumTokens->cesiumIonServerApiUrl ||
            property == pxr::CesiumTokens->cesiumIonServerApplicationId) {
            reloadSession = true;
            updateBindings = true;
        } else if (
            property == pxr::CesiumTokens->cesiumProjectDefaultIonAccessToken ||
            property == pxr::CesiumTokens->cesiumProjectDefaultIonAccessTokenId) {
            updateBindings = true;
        }
    }
    // clang-format on

    if (reloadSession) {
        context.getAssetRegistry().removeIonServer(ionServerPath);
        context.getAssetRegistry().addIonServer(ionServerPath);
    }

    if (updateBindings) {
        updateIonServerBindings(context);
    }
}

void processCesiumCartographicPolygonChanged(
    const Context& context,
    const pxr::SdfPath& cartographicPolygonPath,
    const std::vector<pxr::TfToken>& properties) {

    // Process globe anchor API schema first
    processCesiumGlobeAnchorChanged(context, cartographicPolygonPath, properties);

    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == pxr::UsdTokens->points) {
            updateBindings = true;
        }
    }

    if (updateBindings) {
        updateCartographicPolygonBindings(context, cartographicPolygonPath);
    }
}

void processUsdShaderChanged(
    const Context& context,
    const pxr::SdfPath& shaderPath,
    const std::vector<pxr::TfToken>& properties) {
    const auto usdShader = UsdUtil::getUsdShader(context.getUsdStage(), shaderPath);
    const auto shaderPathFabric = FabricUtil::toFabricPath(shaderPath);
    const auto materialPath = shaderPath.GetParentPath();
    const auto materialPathFabric = FabricUtil::toFabricPath(materialPath);

    if (!UsdUtil::isUsdMaterial(context.getUsdStage(), materialPath)) {
        // Skip if parent path is not a material
        return;
    }

    for (const auto& property : properties) {
        const auto inputNamespace = std::string_view("inputs:");

        const auto& attributeName = property.GetString();

        if (attributeName.rfind(inputNamespace) != 0) {
            // Skip if changed attribute is not a shader input
            return;
        }

        const auto inputName = pxr::TfToken(attributeName.substr(inputNamespace.size()));

        const auto shaderInput = usdShader.GetInput(inputName);
        if (!shaderInput.IsDefined()) {
            // Skip if changed attribute is not a shader input
            return;
        }

        if (shaderInput.HasConnectedSource()) {
            // Skip if shader input is connected to something else
            return;
        }

        if (!FabricUtil::materialHasCesiumNodes(context.getFabricStage(), materialPathFabric)) {
            // Simple materials can be skipped. We only need to handle materials that have been copied to each tile.
            return;
        }

        if (!FabricUtil::isShaderConnectedToMaterial(context.getFabricStage(), materialPathFabric, shaderPathFabric)) {
            // Skip if shader is not connected to the material
            return;
        }

        const auto& tilesets = context.getAssetRegistry().getTilesets();
        for (const auto& pTileset : tilesets) {
            if (pTileset->getMaterialPath() == materialPath) {
                pTileset->updateShaderInput(shaderPath, property);
            }
        }

        context.getFabricResourceManager().updateShaderInput(materialPath, shaderPath, property);
    }
}

[[nodiscard]] bool processCesiumDataRemoved(Context& context, const pxr::SdfPath& dataPath) {
    const auto reloadStage = isFirstData(context, dataPath);
    context.getAssetRegistry().removeData(dataPath);
    return reloadStage;
}

void processCesiumTilesetRemoved(Context& context, const pxr::SdfPath& tilesetPath) {
    context.getAssetRegistry().removeTileset(tilesetPath);
}

void processCesiumIonRasterOverlayRemoved(Context& context, const pxr::SdfPath& ionRasterOverlayPath) {
    context.getAssetRegistry().removeIonRasterOverlay(ionRasterOverlayPath);
    updateRasterOverlayBindings(context, ionRasterOverlayPath);
}

void processCesiumPolygonRasterOverlayRemoved(Context& context, const pxr::SdfPath& polygonRasterOverlayPath) {
    context.getAssetRegistry().removePolygonRasterOverlay(polygonRasterOverlayPath);
    updateRasterOverlayBindings(context, polygonRasterOverlayPath);
}

void processCesiumWebMapServiceRasterOverlayRemoved(
    Context& context,
    const pxr::SdfPath& webMapServiceRasterOverlayPath) {
    context.getAssetRegistry().removeWebMapServiceRasterOverlay(webMapServiceRasterOverlayPath);
    updateRasterOverlayBindings(context, webMapServiceRasterOverlayPath);
}

void processCesiumGeoreferenceRemoved(Context& context, const pxr::SdfPath& georeferencePath) {
    context.getAssetRegistry().removeGeoreference(georeferencePath);
    updateGeoreferenceBindings(context);
}

void processCesiumGlobeAnchorRemoved(Context& context, const pxr::SdfPath& globeAnchorPath) {
    context.getAssetRegistry().removeGlobeAnchor(globeAnchorPath);
    updateGlobeAnchorBindings(context, globeAnchorPath);
}

void processCesiumIonServerRemoved(Context& context, const pxr::SdfPath& ionServerPath) {
    context.getAssetRegistry().removeIonServer(ionServerPath);
    updateIonServerBindings(context);
}

void processCesiumCartographicPolygonRemoved(Context& context, const pxr::SdfPath& cartographicPolygonPath) {
    context.getAssetRegistry().removeCartographicPolygon(cartographicPolygonPath);
    processCesiumGlobeAnchorRemoved(context, cartographicPolygonPath);
    updateCartographicPolygonBindings(context, cartographicPolygonPath);
}

[[nodiscard]] bool processCesiumDataAdded(Context& context, const pxr::SdfPath& dataPath) {
    if (context.getAssetRegistry().getData(dataPath)) {
        return false;
    }

    context.getAssetRegistry().addData(dataPath);
    return isFirstData(context, dataPath);
}

void processCesiumGlobeAnchorAdded(Context& context, const pxr::SdfPath& globeAnchorPath) {
    if (context.getAssetRegistry().getGlobeAnchor(globeAnchorPath)) {
        return;
    }

    context.getAssetRegistry().addGlobeAnchor(globeAnchorPath);
    updateGlobeAnchorBindings(context, globeAnchorPath);
}

void processCesiumTilesetAdded(Context& context, const pxr::SdfPath& tilesetPath) {
    if (UsdUtil::hasCesiumGlobeAnchor(context.getUsdStage(), tilesetPath)) {
        processCesiumGlobeAnchorAdded(context, tilesetPath);
    }

    if (context.getAssetRegistry().getTileset(tilesetPath)) {
        return;
    }

    context.getAssetRegistry().addTileset(tilesetPath);
}

void processCesiumIonRasterOverlayAdded(Context& context, const pxr::SdfPath& ionRasterOverlayPath) {
    if (context.getAssetRegistry().getIonRasterOverlay(ionRasterOverlayPath)) {
        return;
    }

    context.getAssetRegistry().addIonRasterOverlay(ionRasterOverlayPath);
    updateRasterOverlayBindings(context, ionRasterOverlayPath);
}

void processCesiumPolygonRasterOverlayAdded(Context& context, const pxr::SdfPath& polygonRasterOverlayPath) {
    if (context.getAssetRegistry().getPolygonRasterOverlay(polygonRasterOverlayPath)) {
        return;
    }

    context.getAssetRegistry().addPolygonRasterOverlay(polygonRasterOverlayPath);
    updateRasterOverlayBindings(context, polygonRasterOverlayPath);
}

void processCesiumWebMapServiceRasterOverlayAdded(
    Context& context,
    const pxr::SdfPath& webMapServiceRasterOverlayPath) {
    if (context.getAssetRegistry().getWebMapServiceRasterOverlay(webMapServiceRasterOverlayPath)) {
        return;
    }

    context.getAssetRegistry().addWebMapServiceRasterOverlay(webMapServiceRasterOverlayPath);
    updateRasterOverlayBindings(context, webMapServiceRasterOverlayPath);
}

void processCesiumGeoreferenceAdded(Context& context, const pxr::SdfPath& georeferencePath) {
    if (context.getAssetRegistry().getGeoreference(georeferencePath)) {
        return;
    }

    context.getAssetRegistry().addGeoreference(georeferencePath);
    updateGeoreferenceBindings(context);
}

void processCesiumIonServerAdded(Context& context, const pxr::SdfPath& ionServerPath) {
    if (context.getAssetRegistry().getIonServer(ionServerPath)) {
        return;
    }

    context.getAssetRegistry().addIonServer(ionServerPath);
    updateIonServerBindings(context);
}

void processCesiumCartographicPolygonAdded(Context& context, const pxr::SdfPath& cartographicPolygonPath) {
    if (UsdUtil::hasCesiumGlobeAnchor(context.getUsdStage(), cartographicPolygonPath)) {
        processCesiumGlobeAnchorAdded(context, cartographicPolygonPath);
    }

    if (context.getAssetRegistry().getCartographicPolygon(cartographicPolygonPath)) {
        return;
    }

    context.getAssetRegistry().addCartographicPolygon(cartographicPolygonPath);
    updateCartographicPolygonBindings(context, cartographicPolygonPath);
}

} // namespace

UsdNotificationHandler::UsdNotificationHandler(Context* pContext)
    : _pContext(pContext)
    , _noticeListenerKey(
          pxr::TfNotice::Register(pxr::TfCreateWeakPtr(this), &UsdNotificationHandler::onObjectsChanged)) {}

UsdNotificationHandler::~UsdNotificationHandler() {
    pxr::TfNotice::Revoke(_noticeListenerKey);
}

void UsdNotificationHandler::onStageLoaded() {
    // Insert prims manually since USD doesn't notify us about changes when the stage is first loaded
    for (const auto& prim : _pContext->getUsdStage()->Traverse()) {
        const auto type = getTypeFromStage(prim.GetPath());
        if (type != ChangedPrimType::OTHER) {
            insertAddedPrim(prim.GetPath(), type);
        }
    }

    // Process changes immediately
    processChangedPrims();
}

void UsdNotificationHandler::onUpdateFrame() {
    const auto reloadStage = processChangedPrims();

    if (reloadStage) {
        _pContext->reloadStage();
    }
}

void UsdNotificationHandler::clear() {
    _changedPrims.clear();
}

bool UsdNotificationHandler::processChangedPrims() {
    std::vector<ChangedPrim> consolidatedChangedPrims;

    ChangedPrim* pPrevious = nullptr;

    for (const auto& changedPrim : _changedPrims) {
        if (pPrevious && changedPrim.primPath == pPrevious->primPath) {
            if (pPrevious->changedType == ChangedType::PRIM_ADDED &&
                changedPrim.changedType == ChangedType::PROPERTY_CHANGED) {
                // Ignore property changes that occur immediately after the prim is added. This avoids unecessary churn.
                continue;
            }

            if (pPrevious->changedType == ChangedType::PROPERTY_CHANGED &&
                changedPrim.changedType == ChangedType::PROPERTY_CHANGED) {
                // Consolidate property changes so that they can be processed together
                CppUtil::append(pPrevious->properties, changedPrim.properties);
                continue;
            }
        }

        consolidatedChangedPrims.push_back(changedPrim);

        pPrevious = &consolidatedChangedPrims.back();
    }

    _changedPrims.clear();

    auto reloadStage = false;

    for (const auto& changedPrim : consolidatedChangedPrims) {
        reloadStage = processChangedPrim(changedPrim) || reloadStage;
    }

    // Process newly added changes
    if (!_changedPrims.empty()) {
        reloadStage = processChangedPrims() || reloadStage;
    }

    return reloadStage;
}

bool UsdNotificationHandler::processChangedPrim(const ChangedPrim& changedPrim) const {
    auto reloadStage = false;

    switch (changedPrim.changedType) {
        case ChangedType::PROPERTY_CHANGED:
            switch (changedPrim.primType) {
                case ChangedPrimType::CESIUM_DATA:
                    reloadStage = processCesiumDataChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_TILESET:
                    processCesiumTilesetChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_ION_RASTER_OVERLAY:
                    processCesiumIonRasterOverlayChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_POLYGON_RASTER_OVERLAY:
                    processCesiumPolygonRasterOverlayChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_WEB_MAP_SERVICE_RASTER_OVERLAY:
                    processCesiumWebMapServiceRasterOverlayChanged(
                        *_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_GEOREFERENCE:
                    processCesiumGeoreferenceChanged(*_pContext, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                    processCesiumGlobeAnchorChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_ION_SERVER:
                    processCesiumIonServerChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON:
                    processCesiumCartographicPolygonChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::USD_SHADER:
                    processUsdShaderChanged(*_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::OTHER:
                    break;
            }
            break;
        case ChangedType::PRIM_ADDED:
            switch (changedPrim.primType) {
                case ChangedPrimType::CESIUM_DATA:
                    reloadStage = processCesiumDataAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_TILESET:
                    processCesiumTilesetAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_RASTER_OVERLAY:
                    processCesiumIonRasterOverlayAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_POLYGON_RASTER_OVERLAY:
                    processCesiumPolygonRasterOverlayAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_WEB_MAP_SERVICE_RASTER_OVERLAY:
                    processCesiumWebMapServiceRasterOverlayAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GEOREFERENCE:
                    processCesiumGeoreferenceAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                    processCesiumGlobeAnchorAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_SERVER:
                    processCesiumIonServerAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON:
                    processCesiumCartographicPolygonAdded(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::USD_SHADER:
                case ChangedPrimType::OTHER:
                    break;
            }
            break;
        case ChangedType::PRIM_REMOVED:
            switch (changedPrim.primType) {
                case ChangedPrimType::CESIUM_DATA:
                    reloadStage = processCesiumDataRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_TILESET:
                    processCesiumTilesetRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_RASTER_OVERLAY:
                    processCesiumIonRasterOverlayRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_POLYGON_RASTER_OVERLAY:
                    processCesiumPolygonRasterOverlayRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_WEB_MAP_SERVICE_RASTER_OVERLAY:
                    processCesiumWebMapServiceRasterOverlayRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GEOREFERENCE:
                    processCesiumGeoreferenceRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                    processCesiumGlobeAnchorRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_SERVER:
                    processCesiumIonServerRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON:
                    processCesiumCartographicPolygonRemoved(*_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::USD_SHADER:
                case ChangedPrimType::OTHER:
                    break;
            }
            break;
    }

    return reloadStage;
}

void UsdNotificationHandler::onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged) {
    if (!_pContext->hasUsdStage()) {
        return;
    }

    const auto resyncedPaths = objectsChanged.GetResyncedPaths();
    for (const auto& path : resyncedPaths) {
        if (path.IsPrimPath()) {
            if (UsdUtil::primExists(_pContext->getUsdStage(), path)) {
                // A prim is resynced when it is added to the stage or when an API schema is applied to it, e.g. when
                // a material or globe anchor is assigned to a tileset for the first time. We let onPrimAdded get
                // called potentially multiple times so that API schemas can be registered. There are checks later
                // that prevent the prim from being added to the asset registry twice.
                onPrimAdded(path);
            } else {
                onPrimRemoved(path);
            }
        } else if (path.IsPropertyPath()) {
            onPropertyChanged(path);
        }
    }

    const auto changedPaths = objectsChanged.GetChangedInfoOnlyPaths();
    for (const auto& path : changedPaths) {
        if (path.IsPropertyPath()) {
            onPropertyChanged(path);
        }
    }
}

void UsdNotificationHandler::onPrimAdded(const pxr::SdfPath& primPath) {
    const auto type = getTypeFromStage(primPath);
    if (type != ChangedPrimType::OTHER) {
        insertAddedPrim(primPath, type);
    }

    // USD only notifies us about the top-most prim being added. Find all descendant prims
    // and add those as well (recursively)
    const auto prim = _pContext->getUsdStage()->GetPrimAtPath(primPath);
    for (const auto& child : prim.GetAllChildren()) {
        onPrimAdded(child.GetPath());
    }
}

void UsdNotificationHandler::onPrimRemoved(const pxr::SdfPath& primPath) {
    // USD only notifies us about the top-most prim being removed. Find all descendant prims
    // and remove those as well. Since the prims no longer exist on the stage we need
    // to look the paths in _changedPrims and the asset registry.

    // Remove prims that haven't been added to asset registry yet
    // This needs to be an index-based for loop since _changedPrims can grow in size
    const auto changedPrimsCount = _changedPrims.size();
    for (uint64_t i = 0; i < changedPrimsCount; ++i) {
        if (_changedPrims[i].changedType == ChangedType::PRIM_ADDED) {
            if (isPrimOrDescendant(_changedPrims[i].primPath, primPath)) {
                insertRemovedPrim(_changedPrims[i].primPath, _changedPrims[i].primType);
            }
        }
    }

    // Remove prims in the asset registry
    const auto& tilesets = _pContext->getAssetRegistry().getTilesets();
    for (const auto& pTileset : tilesets) {
        const auto tilesetPath = pTileset->getPath();
        if (isPrimOrDescendant(tilesetPath, primPath)) {
            insertRemovedPrim(tilesetPath, ChangedPrimType::CESIUM_TILESET);
        }
    }

    const auto& ionRasterOverlays = _pContext->getAssetRegistry().getIonRasterOverlays();
    for (const auto& pIonRasterOverlay : ionRasterOverlays) {
        const auto ionRasterOverlayPath = pIonRasterOverlay->getPath();
        if (isPrimOrDescendant(ionRasterOverlayPath, primPath)) {
            insertRemovedPrim(ionRasterOverlayPath, ChangedPrimType::CESIUM_ION_RASTER_OVERLAY);
        }
    }

    const auto& polygonRasterOverlays = _pContext->getAssetRegistry().getPolygonRasterOverlays();
    for (const auto& pPolygonRasterOverlay : polygonRasterOverlays) {
        const auto polygonRasterOverlayPath = pPolygonRasterOverlay->getPath();
        if (isPrimOrDescendant(polygonRasterOverlayPath, primPath)) {
            insertRemovedPrim(polygonRasterOverlayPath, ChangedPrimType::CESIUM_POLYGON_RASTER_OVERLAY);
        }
    }

    const auto& georeferences = _pContext->getAssetRegistry().getGeoreferences();
    for (const auto& pGeoreference : georeferences) {
        const auto georeferencePath = pGeoreference->getPath();
        if (isPrimOrDescendant(georeferencePath, primPath)) {
            insertRemovedPrim(georeferencePath, ChangedPrimType::CESIUM_GEOREFERENCE);
        }
    }

    const auto& ionServers = _pContext->getAssetRegistry().getIonServers();
    for (const auto& pIonServer : ionServers) {
        const auto ionServerPath = pIonServer->getPath();
        if (isPrimOrDescendant(ionServerPath, primPath)) {
            insertRemovedPrim(ionServerPath, ChangedPrimType::CESIUM_ION_SERVER);
        }
    }

    const auto& cartographicPolygons = _pContext->getAssetRegistry().getCartographicPolygons();
    for (const auto& pCartographicPolygon : cartographicPolygons) {
        const auto cartographicPolygonPath = pCartographicPolygon->getPath();
        if (isPrimOrDescendant(cartographicPolygonPath, primPath)) {
            insertRemovedPrim(cartographicPolygonPath, ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON);
        }
    }

    const auto& globeAnchors = _pContext->getAssetRegistry().getGlobeAnchors();
    for (const auto& pGlobeAnchor : globeAnchors) {
        const auto globeAnchorPath = pGlobeAnchor->getPath();
        const auto type = getTypeFromAssetRegistry(globeAnchorPath);
        if (type == ChangedPrimType::CESIUM_GLOBE_ANCHOR) {
            // Make sure it's not one of the types previously handled (e.g. cartographic polygon or tileset)
            if (isPrimOrDescendant(globeAnchorPath, primPath)) {
                insertRemovedPrim(globeAnchorPath, ChangedPrimType::CESIUM_GLOBE_ANCHOR);
            }
        }
    }
}

void UsdNotificationHandler::onPropertyChanged(const pxr::SdfPath& propertyPath) {
    const auto& propertyName = propertyPath.GetNameToken();
    const auto primPath = propertyPath.GetPrimPath();
    const auto type = getTypeFromStage(primPath);
    if (type != ChangedPrimType::OTHER) {
        insertPropertyChanged(primPath, type, propertyName);
    }
}

void UsdNotificationHandler::insertAddedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType) {
    // In C++ 20 this can be emplace_back without the {}
    _changedPrims.push_back({primPath, {}, primType, ChangedType::PRIM_ADDED});
}

void UsdNotificationHandler::insertRemovedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType) {
    // In C++ 20 this can be emplace_back without the {}
    _changedPrims.push_back({primPath, {}, primType, ChangedType::PRIM_REMOVED});
}

void UsdNotificationHandler::insertPropertyChanged(
    const pxr::SdfPath& primPath,
    ChangedPrimType primType,
    const pxr::TfToken& propertyName) {
    // In C++ 20 this can be emplace_back without the {}
    _changedPrims.push_back({primPath, {propertyName}, primType, ChangedType::PROPERTY_CHANGED});
}

UsdNotificationHandler::ChangedPrimType UsdNotificationHandler::getTypeFromStage(const pxr::SdfPath& path) const {
    if (UsdUtil::isCesiumData(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_DATA;
    } else if (UsdUtil::isCesiumTileset(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_TILESET;
    } else if (UsdUtil::isCesiumIonRasterOverlay(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_ION_RASTER_OVERLAY;
    } else if (UsdUtil::isCesiumPolygonRasterOverlay(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_POLYGON_RASTER_OVERLAY;
    } else if (UsdUtil::isCesiumWebMapServiceRasterOverlay(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_WEB_MAP_SERVICE_RASTER_OVERLAY;
    } else if (UsdUtil::isCesiumGeoreference(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_GEOREFERENCE;
    } else if (UsdUtil::isCesiumIonServer(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_ION_SERVER;
    } else if (UsdUtil::isCesiumCartographicPolygon(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON;
    } else if (UsdUtil::isUsdShader(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::USD_SHADER;
    } else if (UsdUtil::hasCesiumGlobeAnchor(_pContext->getUsdStage(), path)) {
        // Globe anchor needs to be checked last since prim types take precedence over API schemas
        return ChangedPrimType::CESIUM_GLOBE_ANCHOR;
    }

    return ChangedPrimType::OTHER;
}

UsdNotificationHandler::ChangedPrimType
UsdNotificationHandler::getTypeFromAssetRegistry(const pxr::SdfPath& path) const {
    const auto assetType = _pContext->getAssetRegistry().getAssetType(path);

    switch (assetType) {
        case AssetType::DATA:
            return ChangedPrimType::CESIUM_DATA;
        case AssetType::TILESET:
            return ChangedPrimType::CESIUM_TILESET;
        case AssetType::ION_RASTER_OVERLAY:
            return ChangedPrimType::CESIUM_ION_RASTER_OVERLAY;
        case AssetType::POLYGON_RASTER_OVERLAY:
            return ChangedPrimType::CESIUM_POLYGON_RASTER_OVERLAY;
        case AssetType::WEB_MAP_SERVICE_RASTER_OVERLAY:
            return ChangedPrimType::CESIUM_WEB_MAP_SERVICE_RASTER_OVERLAY;
        case AssetType::GEOREFERENCE:
            return ChangedPrimType::CESIUM_GEOREFERENCE;
        case AssetType::GLOBE_ANCHOR:
            return ChangedPrimType::CESIUM_GLOBE_ANCHOR;
        case AssetType::ION_SERVER:
            return ChangedPrimType::CESIUM_ION_SERVER;
        case AssetType::CARTOGRAPHIC_POLYGON:
            return ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON;
        case AssetType::OTHER:
            return ChangedPrimType::OTHER;
    }

    return ChangedPrimType::OTHER;
}

} // namespace cesium::omniverse
