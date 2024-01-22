#include "cesium/omniverse/UsdNotificationHandler.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/OmniCartographicPolygon.h"
#include "cesium/omniverse/OmniGeoreference.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/OmniIonImagery.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/OmniPolygonImagery.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/tokens.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdShade/shader.h>

namespace cesium::omniverse {

namespace {

bool isPrimOrDescendant(const PXR_NS::SdfPath& descendantPath, const PXR_NS::SdfPath& path) {
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

void updateImageryBindings(Context* pContext, const PXR_NS::SdfPath& imageryPath) {
    const auto& tilesets = pContext->getAssetRegistry().getTilesets();

    // Update tilesets that reference this imagery
    for (const auto& pTileset : tilesets) {
        const auto imageryLayerCount = pTileset->getImageryLayerCount();
        for (uint64_t i = 0; i < imageryLayerCount; ++i) {
            const auto imageryLayerPath = pTileset->getImageryLayerPath(i);
            if (imageryLayerPath == imageryPath) {
                pTileset->reload();
            }
        }
    }
}

void updateImageryBindingsAlpha(Context* pContext, const PXR_NS::SdfPath& imageryPath) {
    const auto& tilesets = pContext->getAssetRegistry().getTilesets();

    // Update tilesets that reference this imagery
    for (const auto& pTileset : tilesets) {
        const auto imageryLayerCount = pTileset->getImageryLayerCount();
        for (uint64_t i = 0; i < imageryLayerCount; ++i) {
            const auto imageryLayerPath = pTileset->getImageryLayerPath(i);
            if (imageryLayerPath == imageryPath) {
                pTileset->updateImageryLayerAlpha(i);
            }
        }
    }
}

void updateIonServerBindings(Context* pContext, const PXR_NS::SdfPath& ionServerPath) {
    // Update tilesets that reference this ion server
    const auto& tilesets = pContext->getAssetRegistry().getTilesets();
    for (const auto& pTileset : tilesets) {
        if (pTileset->getIonServerPath() == ionServerPath) {
            pTileset->reload();
        }
    }

    // Update imageries that reference this ion server
    const auto& ionImageries = pContext->getAssetRegistry().getIonImageries();
    for (const auto& pIonImagery : ionImageries) {
        if (pIonImagery->getIonServerPath() == ionServerPath) {
            pIonImagery->reload();
            updateImageryBindings(pContext, pIonImagery->getPath());
        }
    }
}

void updateCartographicPolygonBindings(Context* pContext, const PXR_NS::SdfPath& cartographicPolygonPath) {
    // Update polygon imageries that reference this cartographic polygon
    const auto& polygonImageries = pContext->getAssetRegistry().getPolygonImageries();
    for (const auto& pPolygonImagery : polygonImageries) {
        const auto paths = pPolygonImagery->getCartographicPolygonPaths();
        if (CppUtil::contains(paths, cartographicPolygonPath)) {
            pPolygonImagery->reload();
            updateImageryBindings(pContext, pPolygonImagery->getPath());
        }
    }
}

void updateGlobeAnchorBindings(Context* pContext, const PXR_NS::SdfPath& globeAnchorPath) {
    if (pContext->getAssetRegistry().getCartographicPolygon(globeAnchorPath)) {
        // Update cartographic polygon that this globe anchor is attached to
        updateCartographicPolygonBindings(pContext, globeAnchorPath);
    }
}

void updateGeoreferenceBindings(Context* pContext, const PXR_NS::SdfPath& georeferencePath) {
    // Don't need to update tilesets. Georeference changes are handled automatically in the update loop.

    // Update globe anchors that reference this georeference
    const auto& globeAnchors = pContext->getAssetRegistry().getGlobeAnchors();
    for (const auto& pGlobeAnchor : globeAnchors) {
        if (pGlobeAnchor->getGeoreferencePath() == georeferencePath) {
            pGlobeAnchor->updateByGeoreference();
            updateGlobeAnchorBindings(pContext, pGlobeAnchor->getPath());
        }
    }
}

bool isFirstData(Context* pContext, const PXR_NS::SdfPath& dataPath) {
    const auto pData = pContext->getAssetRegistry().getData(dataPath);
    const auto pFirstData = pContext->getAssetRegistry().getFirstData();

    return pData && pData == pFirstData;
}

[[nodiscard]] bool processCesiumDataChanged(
    Context* pContext,
    const PXR_NS::SdfPath& dataPath,
    const std::vector<PXR_NS::TfToken>& properties) {

    if (!isFirstData(pContext, dataPath)) {
        return false;
    }

    auto reloadStage = false;
    auto updateGeoreference = false;

    // No change tracking needed for
    // * selectedIonServer
    // * projectDefaultIonAccessToken (deprecated)
    // * projectDefaultIonAccessTokenId (deprecated)

    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumDebugDisableMaterials ||
            property == PXR_NS::CesiumTokens->cesiumDebugDisableTextures ||
            property == PXR_NS::CesiumTokens->cesiumDebugDisableGeometryPool ||
            property == PXR_NS::CesiumTokens->cesiumDebugDisableMaterialPool ||
            property == PXR_NS::CesiumTokens->cesiumDebugDisableTexturePool ||
            property == PXR_NS::CesiumTokens->cesiumDebugGeometryPoolInitialCapacity ||
            property == PXR_NS::CesiumTokens->cesiumDebugMaterialPoolInitialCapacity ||
            property == PXR_NS::CesiumTokens->cesiumDebugTexturePoolInitialCapacity ||
            property == PXR_NS::CesiumTokens->cesiumDebugRandomColors) {
            reloadStage = true;
        } else if (property == PXR_NS::CesiumTokens->cesiumDebugDisableGeoreferencing) {
            updateGeoreference = true;
        }
    }

    if (updateGeoreference) {
        const auto& georeferences = pContext->getAssetRegistry().getGeoreferences();
        for (const auto& pGeoreference : georeferences) {
            updateGeoreferenceBindings(pContext, pGeoreference->getPath());
        }
    }

    return reloadStage;
}

void processCesiumTilesetChanged(
    Context* pContext,
    const PXR_NS::SdfPath& tilesetPath,
    const std::vector<PXR_NS::TfToken>& properties) {
    const auto pTileset = pContext->getAssetRegistry().getTileset(tilesetPath);
    if (!pTileset) {
        return;
    }

    auto reload = false;
    auto updateTilesetOptions = false;
    auto updateDisplayColorAndOpacity = false;

    // No change tracking needed for
    // * suspendUpdate
    // * georeferenceBinding
    // * Transform changes (handled automatically in update loop)

    // clang-format off
    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumSourceType ||
            property == PXR_NS::CesiumTokens->cesiumUrl ||
            property == PXR_NS::CesiumTokens->cesiumIonAssetId ||
            property == PXR_NS::CesiumTokens->cesiumIonAccessToken ||
            property == PXR_NS::CesiumTokens->cesiumIonServerBinding ||
            property == PXR_NS::CesiumTokens->cesiumSmoothNormals ||
            property == PXR_NS::CesiumTokens->cesiumShowCreditsOnScreen ||
            property == PXR_NS::UsdTokens->material_binding) {
            reload = true;
        } else if (
            property == PXR_NS::CesiumTokens->cesiumMaximumScreenSpaceError ||
            property == PXR_NS::CesiumTokens->cesiumPreloadAncestors ||
            property == PXR_NS::CesiumTokens->cesiumPreloadSiblings ||
            property == PXR_NS::CesiumTokens->cesiumForbidHoles ||
            property == PXR_NS::CesiumTokens->cesiumMaximumSimultaneousTileLoads ||
            property == PXR_NS::CesiumTokens->cesiumMaximumCachedBytes ||
            property == PXR_NS::CesiumTokens->cesiumLoadingDescendantLimit ||
            property == PXR_NS::CesiumTokens->cesiumEnableFrustumCulling ||
            property == PXR_NS::CesiumTokens->cesiumEnableFogCulling ||
            property == PXR_NS::CesiumTokens->cesiumEnforceCulledScreenSpaceError ||
            property == PXR_NS::CesiumTokens->cesiumCulledScreenSpaceError ||
            property == PXR_NS::CesiumTokens->cesiumMainThreadLoadingTimeLimit) {
            updateTilesetOptions = true;
        } else if (
            property == PXR_NS::UsdTokens->primvars_displayColor ||
            property == PXR_NS::UsdTokens->primvars_displayOpacity) {
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

void processCesiumImageryChanged(
    Context* pContext,
    const PXR_NS::SdfPath& imageryPath,
    const std::vector<PXR_NS::TfToken>& properties) {
    const auto pImagery = pContext->getAssetRegistry().getImagery(imageryPath);
    if (!pImagery) {
        return;
    }

    auto reload = false;
    auto updateBindings = false;
    auto updateImageryLayerAlpha = false;

    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumShowCreditsOnScreen) {
            reload = true;
            updateBindings = true;
        } else if (property == PXR_NS::CesiumTokens->cesiumOverlayRenderMethod) {
            updateBindings = true;
        } else if (property == PXR_NS::CesiumTokens->cesiumAlpha) {
            updateImageryLayerAlpha = true;
        }
    }

    if (reload) {
        pImagery->reload();
    }

    if (updateBindings) {
        updateImageryBindings(pContext, imageryPath);
    }

    if (updateImageryLayerAlpha) {
        updateImageryBindingsAlpha(pContext, imageryPath);
    }
}

void processCesiumIonImageryChanged(
    Context* pContext,
    const PXR_NS::SdfPath& ionImageryPath,
    const std::vector<PXR_NS::TfToken>& properties) {
    const auto pIonImagery = pContext->getAssetRegistry().getIonImagery(ionImageryPath);
    if (!pIonImagery) {
        return;
    }

    // Process base class first
    processCesiumImageryChanged(pContext, ionImageryPath, properties);

    auto reload = false;
    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumIonAssetId ||
            property == PXR_NS::CesiumTokens->cesiumIonAccessToken ||
            property == PXR_NS::CesiumTokens->cesiumIonServerBinding) {
            reload = true;
            updateBindings = true;
        }
    }

    if (reload) {
        pIonImagery->reload();
    }

    if (updateBindings) {
        updateImageryBindings(pContext, ionImageryPath);
    }
}

void processCesiumPolygonImageryChanged(
    Context* pContext,
    const PXR_NS::SdfPath& polygonImageryPath,
    const std::vector<PXR_NS::TfToken>& properties) {
    const auto pPolygonImagery = pContext->getAssetRegistry().getPolygonImagery(polygonImageryPath);
    if (!pPolygonImagery) {
        return;
    }

    // Process base class first
    processCesiumImageryChanged(pContext, polygonImageryPath, properties);

    auto reload = false;
    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumCartographicPolygonBinding) {
            reload = true;
            updateBindings = true;
        }
    }

    if (reload) {
        pPolygonImagery->reload();
    }

    if (updateBindings) {
        updateImageryBindings(pContext, polygonImageryPath);
    }
}

void processCesiumGeoreferenceChanged(
    Context* pContext,
    const PXR_NS::SdfPath& georeferencePath,
    const std::vector<PXR_NS::TfToken>& properties) {

    auto updateBindings = false;

    // clang-format off
    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumGeoreferenceOriginLongitude ||
            property == PXR_NS::CesiumTokens->cesiumGeoreferenceOriginLatitude ||
            property == PXR_NS::CesiumTokens->cesiumGeoreferenceOriginHeight) {
            updateBindings = true;
        }
    }
    // clang-format on

    if (updateBindings) {
        updateGeoreferenceBindings(pContext, georeferencePath);
    }
}

void processCesiumGlobeAnchorChanged(
    Context* pContext,
    const PXR_NS::SdfPath& globeAnchorPath,
    const std::vector<PXR_NS::TfToken>& properties) {
    const auto pGlobeAnchor = pContext->getAssetRegistry().getGlobeAnchor(globeAnchorPath);
    if (!pGlobeAnchor) {
        return;
    }

    auto updateByPrimLocalTransform = false;
    auto updateByGeographicCoordinates = false;
    auto updateByPrimLocalToEcefTransform = false;
    auto updateByGeoreference = false;
    auto updateBindings = false;

    const auto detectTransformChanges = pGlobeAnchor->getDetectTransformChanges();
    const auto adjustOrientation = pGlobeAnchor->getAdjustOrientation();

    for (const auto& property : properties) {
        if (detectTransformChanges &&
            (property == PXR_NS::CesiumTokens->cesiumAnchorDetectTransformChanges ||
             property == PXR_NS::UsdTokens->xformOp_translate || property == PXR_NS::UsdTokens->xformOp_rotateXYZ ||
             property == PXR_NS::UsdTokens->xformOp_scale)) {
            updateByPrimLocalTransform = true;
            updateBindings = true;
        } else if (property == PXR_NS::CesiumTokens->cesiumAnchorGeographicCoordinates) {
            updateByGeographicCoordinates = true;
            updateBindings = true;
        } else if (
            (adjustOrientation && property == PXR_NS::CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving) ||
            property == PXR_NS::CesiumTokens->cesiumAnchorPosition ||
            property == PXR_NS::CesiumTokens->cesiumAnchorRotation ||
            property == PXR_NS::CesiumTokens->cesiumAnchorScale) {
            updateByPrimLocalToEcefTransform = true;
            updateBindings = true;
        } else if (property == PXR_NS::CesiumTokens->cesiumAnchorGeoreferenceBinding) {
            updateByGeoreference = true;
            updateBindings = true;
        }
    }

    if (updateByPrimLocalTransform) {
        pGlobeAnchor->updateByPrimLocalTransform();
    }

    if (updateByGeographicCoordinates) {
        pGlobeAnchor->updateByGeographicCoordinates();
    }

    if (updateByPrimLocalToEcefTransform) {
        pGlobeAnchor->updateByPrimLocalToEcefTransform();
    }

    if (updateByGeoreference) {
        pGlobeAnchor->updateByGeoreference();
    }

    if (updateBindings) {
        updateGlobeAnchorBindings(pContext, globeAnchorPath);
    }
}

void processCesiumIonServerChanged(
    Context* pContext,
    const PXR_NS::SdfPath& ionServerPath,
    const std::vector<PXR_NS::TfToken>& properties) {

    auto reloadSession = false;
    auto updateBindings = false;

    // No change tracking needed for
    // * displayName

    // clang-format off
    for (const auto& property : properties) {
        if (property == PXR_NS::CesiumTokens->cesiumIonServerUrl ||
            property == PXR_NS::CesiumTokens->cesiumIonServerApiUrl ||
            property == PXR_NS::CesiumTokens->cesiumIonServerApplicationId) {
            reloadSession = true;
            updateBindings = true;
        } else if (
            property == PXR_NS::CesiumTokens->cesiumProjectDefaultIonAccessToken ||
            property == PXR_NS::CesiumTokens->cesiumProjectDefaultIonAccessTokenId) {
            updateBindings = true;
        }
    }
    // clang-format on

    if (reloadSession) {
        pContext->getAssetRegistry().removeIonServer(ionServerPath);
        pContext->getAssetRegistry().addIonServer(ionServerPath);
    }

    if (updateBindings) {
        updateIonServerBindings(pContext, ionServerPath);
    }
}

void processCesiumCartographicPolygonChanged(
    Context* pContext,
    const PXR_NS::SdfPath& cartographicPolygonPath,
    const std::vector<PXR_NS::TfToken>& properties) {

    // Process globe anchor API schema first
    processCesiumGlobeAnchorChanged(pContext, cartographicPolygonPath, properties);

    auto updateBindings = false;

    for (const auto& property : properties) {
        if (property == PXR_NS::UsdTokens->points) {
            updateBindings = true;
        }
    }

    if (updateBindings) {
        updateCartographicPolygonBindings(pContext, cartographicPolygonPath);
    }
}

void processUsdShaderChanged(
    Context* pContext,
    const PXR_NS::SdfPath& shaderPath,
    const std::vector<PXR_NS::TfToken>& properties) {
    const auto usdShader = UsdUtil::getUsdShader(pContext->getUsdStage(), shaderPath);
    const auto shaderPathFabric = FabricUtil::toFabricPath(shaderPath);
    const auto materialPath = shaderPath.GetParentPath();
    const auto materialPathFabric = FabricUtil::toFabricPath(materialPath);

    if (!UsdUtil::isUsdMaterial(pContext->getUsdStage(), materialPath)) {
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

        const auto inputName = PXR_NS::TfToken(attributeName.substr(inputNamespace.size()));

        const auto shaderInput = usdShader.GetInput(inputName);
        if (!shaderInput.IsDefined()) {
            // Skip if changed attribute is not a shader input
            return;
        }

        if (shaderInput.HasConnectedSource()) {
            // Skip if shader input is connected to something else
            return;
        }

        if (!FabricUtil::materialHasCesiumNodes(pContext->getFabricStage(), materialPathFabric)) {
            // Simple materials can be skipped. We only need to handle materials that have been copied to each tile.
            return;
        }

        if (!FabricUtil::isShaderConnectedToMaterial(
                pContext->getFabricStage(), materialPathFabric, shaderPathFabric)) {
            // Skip if shader is not connected to the material
            return;
        }

        const auto& tilesets = pContext->getAssetRegistry().getTilesets();
        for (const auto& pTileset : tilesets) {
            if (pTileset->getMaterialPath() == materialPath) {
                pTileset->updateShaderInput(shaderPath, property);
            }
        }

        pContext->getFabricResourceManager().updateShaderInput(materialPath, shaderPath, property);
    }
}

[[nodiscard]] bool processCesiumDataRemoved(Context* pContext, const PXR_NS::SdfPath& dataPath) {
    const auto reloadStage = isFirstData(pContext, dataPath);
    pContext->getAssetRegistry().removeData(dataPath);
    return reloadStage;
}

void processCesiumTilesetRemoved(Context* pContext, const PXR_NS::SdfPath& tilesetPath) {
    pContext->getAssetRegistry().removeTileset(tilesetPath);
}

void processCesiumIonImageryRemoved(Context* pContext, const PXR_NS::SdfPath& ionImageryPath) {
    pContext->getAssetRegistry().removeIonImagery(ionImageryPath);
    updateImageryBindings(pContext, ionImageryPath);
}

void processCesiumPolygonImageryRemoved(Context* pContext, const PXR_NS::SdfPath& polygonImageryPath) {
    pContext->getAssetRegistry().removePolygonImagery(polygonImageryPath);
    updateImageryBindings(pContext, polygonImageryPath);
}

void processCesiumGeoreferenceRemoved(Context* pContext, const PXR_NS::SdfPath& georeferencePath) {
    pContext->getAssetRegistry().removeGeoreference(georeferencePath);
    updateGeoreferenceBindings(pContext, georeferencePath);
}

void processCesiumGlobeAnchorRemoved(Context* pContext, const PXR_NS::SdfPath& globeAnchorPath) {
    pContext->getAssetRegistry().removeGlobeAnchor(globeAnchorPath);
    updateGlobeAnchorBindings(pContext, globeAnchorPath);
}

void processCesiumIonServerRemoved(Context* pContext, const PXR_NS::SdfPath& ionServerPath) {
    pContext->getAssetRegistry().removeIonServer(ionServerPath);
    updateIonServerBindings(pContext, ionServerPath);
}

void processCesiumCartographicPolygonRemoved(Context* pContext, const PXR_NS::SdfPath& cartographicPolygonPath) {
    pContext->getAssetRegistry().removeCartographicPolygon(cartographicPolygonPath);
    processCesiumGlobeAnchorRemoved(pContext, cartographicPolygonPath);
    updateCartographicPolygonBindings(pContext, cartographicPolygonPath);
}

[[nodiscard]] bool processCesiumDataAdded(Context* pContext, const PXR_NS::SdfPath& dataPath) {
    pContext->getAssetRegistry().addData(dataPath);
    return isFirstData(pContext, dataPath);
}

void processCesiumTilesetAdded(Context* pContext, const PXR_NS::SdfPath& tilesetPath) {
    pContext->getAssetRegistry().addTileset(tilesetPath);
}

void processCesiumIonImageryAdded(Context* pContext, const PXR_NS::SdfPath& ionImageryPath) {
    pContext->getAssetRegistry().addIonImagery(ionImageryPath);
    updateImageryBindings(pContext, ionImageryPath);
}

void processCesiumPolygonImageryAdded(Context* pContext, const PXR_NS::SdfPath& polygonImageryPath) {
    pContext->getAssetRegistry().addPolygonImagery(polygonImageryPath);
    updateImageryBindings(pContext, polygonImageryPath);
}

void processCesiumGeoreferenceAdded(Context* pContext, const PXR_NS::SdfPath& georeferencePath) {
    pContext->getAssetRegistry().addGeoreference(georeferencePath);
    updateGeoreferenceBindings(pContext, georeferencePath);
}

void processCesiumGlobeAnchorAdded(Context* pContext, const PXR_NS::SdfPath& globeAnchorPath) {
    pContext->getAssetRegistry().addGlobeAnchor(globeAnchorPath);
    updateGlobeAnchorBindings(pContext, globeAnchorPath);
}

void processCesiumIonServerAdded(Context* pContext, const PXR_NS::SdfPath& ionServerPath) {
    pContext->getAssetRegistry().addIonServer(ionServerPath);
    updateIonServerBindings(pContext, ionServerPath);
}

void processCesiumCartographicPolygonAdded(Context* pContext, const PXR_NS::SdfPath& cartographicPolygonPath) {
    processCesiumGlobeAnchorAdded(pContext, cartographicPolygonPath);
    pContext->getAssetRegistry().addCartographicPolygon(cartographicPolygonPath);
    updateCartographicPolygonBindings(pContext, cartographicPolygonPath);
}

} // namespace

UsdNotificationHandler::UsdNotificationHandler(Context* pContext)
    : _pContext(pContext)
    , _noticeListenerKey(
          PXR_NS::TfNotice::Register(PXR_NS::TfCreateWeakPtr(this), &UsdNotificationHandler::onObjectsChanged)) {}

UsdNotificationHandler::~UsdNotificationHandler() {
    PXR_NS::TfNotice::Revoke(_noticeListenerKey);
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
                    reloadStage = processCesiumDataChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_TILESET:
                    processCesiumTilesetChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_ION_IMAGERY:
                    processCesiumIonImageryChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_POLYGON_IMAGERY:
                    processCesiumPolygonImageryChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_GEOREFERENCE:
                    processCesiumGeoreferenceChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                    processCesiumGlobeAnchorChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_ION_SERVER:
                    processCesiumIonServerChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON:
                    processCesiumCartographicPolygonChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::USD_SHADER:
                    processUsdShaderChanged(_pContext, changedPrim.primPath, changedPrim.properties);
                    break;
                case ChangedPrimType::OTHER:
                    break;
            }
            break;
        case ChangedType::PRIM_ADDED:
            switch (changedPrim.primType) {
                case ChangedPrimType::CESIUM_DATA:
                    reloadStage = processCesiumDataAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_TILESET:
                    processCesiumTilesetAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_IMAGERY:
                    processCesiumIonImageryAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_POLYGON_IMAGERY:
                    processCesiumPolygonImageryAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GEOREFERENCE:
                    processCesiumGeoreferenceAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                    processCesiumGlobeAnchorAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_SERVER:
                    processCesiumIonServerAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON:
                    processCesiumCartographicPolygonAdded(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::USD_SHADER:
                case ChangedPrimType::OTHER:
                    break;
            }
            break;
        case ChangedType::PRIM_REMOVED:
            switch (changedPrim.primType) {
                case ChangedPrimType::CESIUM_DATA:
                    reloadStage = processCesiumDataRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_TILESET:
                    processCesiumTilesetRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_IMAGERY:
                    processCesiumIonImageryRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_POLYGON_IMAGERY:
                    processCesiumPolygonImageryRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GEOREFERENCE:
                    processCesiumGeoreferenceRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                    processCesiumGlobeAnchorRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_ION_SERVER:
                    processCesiumIonServerRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::CESIUM_CARTOGRAPHIC_POLYGON:
                    processCesiumCartographicPolygonRemoved(_pContext, changedPrim.primPath);
                    break;
                case ChangedPrimType::USD_SHADER:
                case ChangedPrimType::OTHER:
                    break;
            }
            break;
    }

    return reloadStage;
}

bool UsdNotificationHandler::alreadyRegistered(const PXR_NS::SdfPath& path) {
    const auto alreadyAdded = [&path](const auto& changedPrim) {
        return changedPrim.primPath == path && changedPrim.changedType == ChangedType::PRIM_ADDED;
    };

    if (CppUtil::containsIf(_changedPrims, alreadyAdded)) {
        return true;
    }

    return _pContext->getAssetRegistry().hasAsset(path);
}

void UsdNotificationHandler::onObjectsChanged(const PXR_NS::UsdNotice::ObjectsChanged& objectsChanged) {
    if (!_pContext->hasUsdStage()) {
        return;
    }

    const auto resyncedPaths = objectsChanged.GetResyncedPaths();
    for (const auto& path : resyncedPaths) {
        if (path.IsPrimPath()) {
            if (UsdUtil::primExists(_pContext->getUsdStage(), path)) {
                if (alreadyRegistered(path)) {
                    // A prim may be resynced even if its path doesn't change, like when an API schema is applied to
                    //it, e.g. when a material is assigned to a tileset for the first time. Do nothing for now. In the
                    // future if we support attaching globe anchors to tilesets this will have to change so that the
                    // notification doesn't get lost.
                    continue;
                }

                onPrimAdded(path);
            } else {
                onPrimRemoved(path);
            }
        }
    }

    const auto changedPaths = objectsChanged.GetChangedInfoOnlyPaths();
    for (const auto& path : changedPaths) {
        if (path.IsPropertyPath()) {
            onPropertyChanged(path);
        }
    }
}

void UsdNotificationHandler::onPrimAdded(const PXR_NS::SdfPath& primPath) {
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

void UsdNotificationHandler::onPrimRemoved(const PXR_NS::SdfPath& primPath) {
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

    const auto& ionImageries = _pContext->getAssetRegistry().getIonImageries();
    for (const auto& pIonImagery : ionImageries) {
        const auto ionImageryPath = pIonImagery->getPath();
        if (isPrimOrDescendant(ionImageryPath, primPath)) {
            insertRemovedPrim(ionImageryPath, ChangedPrimType::CESIUM_ION_IMAGERY);
        }
    }

    const auto& polygonImageries = _pContext->getAssetRegistry().getPolygonImageries();
    for (const auto& pPolygonImagery : polygonImageries) {
        const auto polygonImageryPath = pPolygonImagery->getPath();
        if (isPrimOrDescendant(polygonImageryPath, primPath)) {
            insertRemovedPrim(polygonImageryPath, ChangedPrimType::CESIUM_POLYGON_IMAGERY);
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
            // Make sure it's not one of the types previously handled (e.g. cartographic polygon)
            if (isPrimOrDescendant(globeAnchorPath, primPath)) {
                insertRemovedPrim(globeAnchorPath, ChangedPrimType::CESIUM_GLOBE_ANCHOR);
            }
        }
    }
}

void UsdNotificationHandler::onPropertyChanged(const PXR_NS::SdfPath& propertyPath) {
    const auto& propertyName = propertyPath.GetNameToken();
    const auto primPath = propertyPath.GetPrimPath();
    const auto type = getTypeFromStage(primPath);
    if (type != ChangedPrimType::OTHER) {
        insertPropertyChanged(primPath, type, propertyName);
    }
}

void UsdNotificationHandler::insertAddedPrim(const PXR_NS::SdfPath& primPath, ChangedPrimType primType) {
    // In C++ 20 this can be emplace_back without the {}
    _changedPrims.push_back({primPath, {}, primType, ChangedType::PRIM_ADDED});
}

void UsdNotificationHandler::insertRemovedPrim(const PXR_NS::SdfPath& primPath, ChangedPrimType primType) {
    // In C++ 20 this can be emplace_back without the {}
    _changedPrims.push_back({primPath, {}, primType, ChangedType::PRIM_REMOVED});
}

void UsdNotificationHandler::insertPropertyChanged(
    const PXR_NS::SdfPath& primPath,
    ChangedPrimType primType,
    const PXR_NS::TfToken& propertyName) {
    // In C++ 20 this can be emplace_back without the {}
    _changedPrims.push_back({primPath, {propertyName}, primType, ChangedType::PROPERTY_CHANGED});
}

UsdNotificationHandler::ChangedPrimType UsdNotificationHandler::getTypeFromStage(const PXR_NS::SdfPath& path) const {
    if (UsdUtil::isCesiumData(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_DATA;
    } else if (UsdUtil::isCesiumTileset(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_TILESET;
    } else if (UsdUtil::isCesiumIonImagery(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_ION_IMAGERY;
    } else if (UsdUtil::isCesiumPolygonImagery(_pContext->getUsdStage(), path)) {
        return ChangedPrimType::CESIUM_POLYGON_IMAGERY;
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
UsdNotificationHandler::getTypeFromAssetRegistry(const PXR_NS::SdfPath& path) const {
    const auto assetType = _pContext->getAssetRegistry().getAssetType(path);

    switch (assetType) {
        case AssetType::DATA:
            return ChangedPrimType::CESIUM_DATA;
        case AssetType::TILESET:
            return ChangedPrimType::CESIUM_TILESET;
        case AssetType::ION_IMAGERY:
            return ChangedPrimType::CESIUM_ION_IMAGERY;
        case AssetType::POLYGON_IMAGERY:
            return ChangedPrimType::CESIUM_POLYGON_IMAGERY;
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
