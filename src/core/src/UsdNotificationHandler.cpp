#include "cesium/omniverse/UsdNotificationHandler.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GeospatialUtil.h"
#include "cesium/omniverse/GlobeAnchorRegistry.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/OmniGlobeAnchor.h"
#include "cesium/omniverse/OmniImagery.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/SessionRegistry.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <pxr/usd/usd/primRange.h>

namespace cesium::omniverse {

namespace {
PrimType getType(const pxr::SdfPath& path) {
    if (UsdUtil::primExists(path)) {
        if (UsdUtil::isCesiumData(path)) {
            return PrimType::CESIUM_DATA;
        } else if (UsdUtil::isCesiumTileset(path)) {
            return PrimType::CESIUM_TILESET;
        } else if (UsdUtil::isCesiumImagery(path)) {
            return PrimType::CESIUM_IMAGERY;
        } else if (UsdUtil::isCesiumGeoreference(path)) {
            return PrimType::CESIUM_GEOREFERENCE;
        } else if (UsdUtil::hasCesiumGlobeAnchor(path)) {
            return PrimType::CESIUM_GLOBE_ANCHOR;
        } else if (UsdUtil::isCesiumIonServer(path)) {
            return PrimType::CESIUM_ION_SERVER;
        } else if (UsdUtil::isUsdShader(path)) {
            return PrimType::USD_SHADER;
        }
    } else {
        // If the prim doesn't exist, because it was removed from the stage already, we can get the type from the asset registry
        const auto assetType = AssetRegistry::getInstance().getAssetType(path);

        switch (assetType) {
            case AssetType::TILESET:
                return PrimType::CESIUM_TILESET;
            case AssetType::IMAGERY:
                return PrimType::CESIUM_IMAGERY;
            default:
                break;
        }

        // If we still haven't found the prim type, it could be a globe anchor, and we should check if it exists in the globe anchor registry
        if (GlobeAnchorRegistry::getInstance().anchorExists(path)) {
            return PrimType::CESIUM_GLOBE_ANCHOR;
        }

        // If we still haven't found the prim type, it could be a Cesium ion session, and we should check if it exists in the session registry
        if (SessionRegistry::getInstance().sessionExists(path)) {
            return PrimType::CESIUM_ION_SERVER;
        }
    }

    return PrimType::OTHER;
}

bool inSubtree(const pxr::SdfPath& path, const pxr::SdfPath& descendantPath) {
    if (path == descendantPath) {
        return true;
    }

    for (const auto& ancestorPath : descendantPath.GetAncestorsRange()) {
        if (ancestorPath == path) {
            return true;
        }
    }

    return false;
}

} // namespace

UsdNotificationHandler::UsdNotificationHandler() {
    _noticeListenerKey = pxr::TfNotice::Register(pxr::TfCreateWeakPtr(this), &UsdNotificationHandler::onObjectsChanged);
}

UsdNotificationHandler::~UsdNotificationHandler() {
    pxr::TfNotice::Revoke(_noticeListenerKey);
}

void UsdNotificationHandler::onStageLoaded() {
    const auto stage = UsdUtil::getUsdStage();

    // Add sessions first since they can be referenced by tilesets and imagery layers
    for (const auto& prim : stage->Traverse()) {
        const auto& path = prim.GetPath();
        if (getType(path) == PrimType::CESIUM_ION_SERVER) {
            insertAddedPrim(path, PrimType::CESIUM_ION_SERVER);
        }
    }

    for (const auto& prim : stage->Traverse()) {
        const auto& path = prim.GetPath();
        if (UsdUtil::isCesiumTileset(path)) {
            insertAddedPrim(path, PrimType::CESIUM_TILESET);
        } else if (UsdUtil::isCesiumImagery(path)) {
            insertAddedPrim(path, PrimType::CESIUM_IMAGERY);
        } else if (UsdUtil::hasCesiumGlobeAnchor(path)) {
            insertAddedPrim(path, PrimType::CESIUM_GLOBE_ANCHOR);
        }
    }
}

void UsdNotificationHandler::onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged) {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto& resyncedPaths = objectsChanged.GetResyncedPaths();
    for (const auto& path : resyncedPaths) {
        if (path.IsPrimPath()) {
            if (UsdUtil::primExists(path)) {
                const auto isTileset = getType(path) == PrimType::CESIUM_TILESET;
                const auto isTilesetAlreadyRegistered = AssetRegistry::getInstance().getTilesetByPath(path).has_value();

                if (isTileset && isTilesetAlreadyRegistered) {
                    // A prim may be resynced even if its path doesn't change, like when an API Schema is applied to it.
                    // This happens when a material is assigned to a tileset for the first time.
                    // We don't want to add the prim again if it's already registered.
                    continue;
                }

                onPrimAdded(path);
            } else {
                onPrimRemoved(path);
            }
        }
    }

    const auto& changedPaths = objectsChanged.GetChangedInfoOnlyPaths();
    for (const auto& path : changedPaths) {
        if (path.IsPropertyPath()) {
            onPropertyChanged(path);
        }
    }
}

void UsdNotificationHandler::onPrimAdded(const pxr::SdfPath& primPath) {
    const auto type = getType(primPath);
    if (type != PrimType::OTHER) {
        insertAddedPrim(primPath, type);
    }

    // USD only notifies us about the top-most prim. Traverse over descendant prims and add those as well.
    // This comes up when a tileset with imagery is moved or renamed.
    const auto stage = UsdUtil::getUsdStage();
    const auto prim = stage->GetPrimAtPath(primPath);
    for (const auto& child : prim.GetAllChildren()) {
        onPrimAdded(child.GetPath());
    }
}

void UsdNotificationHandler::onPrimRemoved(const pxr::SdfPath& primPath) {
    // USD only notifies us about the top-most prim. This prim may have tileset / imagery descendants that need to
    // be removed as well. Unlike onPrimAdded we can't traverse the stage because these prims no longer exist. Instead
    // loop through tilesets and imagery in the asset registry.
    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        const auto tilesetPath = tileset->getPath();
        const auto type = getType(tilesetPath);
        if (type == PrimType::CESIUM_TILESET) {
            if (inSubtree(primPath, tilesetPath)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }

    const auto& imageries = AssetRegistry::getInstance().getAllImageries();
    for (const auto& imagery : imageries) {
        const auto imageryPath = imagery->getPath();
        const auto type = getType(imageryPath);
        if (type == PrimType::CESIUM_IMAGERY) {
            if (inSubtree(primPath, imageryPath)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }

    const auto& anchors = GlobeAnchorRegistry::getInstance().getAllAnchorPaths();
    for (const auto& anchorPath : anchors) {
        const auto& path = pxr::SdfPath(anchorPath);
        const auto& type = getType(path);
        if (type == PrimType::CESIUM_GLOBE_ANCHOR) {
            if (inSubtree(primPath, path)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }

    const auto& sessions = SessionRegistry::getInstance().getAllSessionPaths();
    for (const auto& path : sessions) {
        const auto& type = getType(path);
        if (type == PrimType::CESIUM_ION_SERVER) {
            if (inSubtree(primPath, path)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }
}

void UsdNotificationHandler::onPropertyChanged(const pxr::SdfPath& propertyPath) {
    const auto& propertyName = propertyPath.GetNameToken();
    const auto& primPath = propertyPath.GetPrimPath();
    const auto& type = getType(primPath);
    if (type != PrimType::OTHER) {
        insertPropertyChangedPrim(primPath, type, propertyName);
    }
}

void UsdNotificationHandler::insertAddedPrim(const pxr::SdfPath& primPath, PrimType primType) {
    _addedPrims.emplace_back(AddedPrim{primPath, primType});
    CESIUM_LOG_TRACE("Added prim: {}", primPath.GetText());
}

void UsdNotificationHandler::insertRemovedPrim(const pxr::SdfPath& primPath, PrimType primType) {
    _removedPrims.emplace_back(RemovedPrim{primPath, primType});
    CESIUM_LOG_TRACE("Removed prim: {}", primPath.GetText());
}

void UsdNotificationHandler::insertPropertyChangedPrim(
    const pxr::SdfPath& primPath,
    PrimType primType,
    const pxr::TfToken& propertyName) {
    _propertyChangedPrims.emplace_back(PropertyChangedPrim{primPath, primType, propertyName});
    CESIUM_LOG_TRACE("Property changed: {} {}", primPath.GetText(), propertyName.GetText());
}

namespace {

void reloadIonServerAssets(const pxr::SdfPath& ionServerPath) {
    // Reload tilesets that reference this ion server
    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        if (tileset->getIonServerPath() == ionServerPath) {
            tileset->reload();
        }
    }

    // Reload tilesets whose imagery layers reference this ion server
    const auto& imageries = AssetRegistry::getInstance().getAllImageries();
    for (const auto& imagery : imageries) {
        if (imagery->getIonServerPath() == ionServerPath) {
            const auto tilesetPath = imagery->getPath().GetParentPath();
            const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);
            if (tileset.has_value()) {
                tileset.value()->reload();
            }
        }
    }
}

void processCesiumDataChanged(const pxr::TfToken& propertyName) {
    if (propertyName == pxr::CesiumTokens->cesiumDebugDisableMaterials ||
        propertyName == pxr::CesiumTokens->cesiumDebugDisableTextures ||
        propertyName == pxr::CesiumTokens->cesiumDebugDisableGeometryPool ||
        propertyName == pxr::CesiumTokens->cesiumDebugDisableMaterialPool ||
        propertyName == pxr::CesiumTokens->cesiumDebugDisableTexturePool ||
        propertyName == pxr::CesiumTokens->cesiumDebugGeometryPoolInitialCapacity ||
        propertyName == pxr::CesiumTokens->cesiumDebugMaterialPoolInitialCapacity ||
        propertyName == pxr::CesiumTokens->cesiumDebugTexturePoolInitialCapacity ||
        propertyName == pxr::CesiumTokens->cesiumDebugRandomColors) {
        Context::instance().reloadStage();
    }
}

void processCesiumTilesetChanged(const pxr::SdfPath& tilesetPath, const pxr::TfToken& propertyName) {
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);
    if (!tileset.has_value()) {
        return;
    }

    if (propertyName == pxr::CesiumTokens->cesiumMaximumScreenSpaceError ||
        propertyName == pxr::CesiumTokens->cesiumPreloadAncestors ||
        propertyName == pxr::CesiumTokens->cesiumPreloadSiblings ||
        propertyName == pxr::CesiumTokens->cesiumForbidHoles ||
        propertyName == pxr::CesiumTokens->cesiumMaximumSimultaneousTileLoads ||
        propertyName == pxr::CesiumTokens->cesiumMaximumCachedBytes ||
        propertyName == pxr::CesiumTokens->cesiumLoadingDescendantLimit ||
        propertyName == pxr::CesiumTokens->cesiumEnableFrustumCulling ||
        propertyName == pxr::CesiumTokens->cesiumEnableFogCulling ||
        propertyName == pxr::CesiumTokens->cesiumEnforceCulledScreenSpaceError ||
        propertyName == pxr::CesiumTokens->cesiumCulledScreenSpaceError ||
        propertyName == pxr::CesiumTokens->cesiumMainThreadLoadingTimeLimit) {
        // Update tileset options
        tileset.value()->updateTilesetOptions();
    } else if (
        propertyName == pxr::UsdTokens->primvars_displayColor ||
        propertyName == pxr::UsdTokens->primvars_displayOpacity) {
        // Update display color and opacity
        tileset.value()->updateDisplayColorAndOpacity();
    } else if (
        propertyName == pxr::CesiumTokens->cesiumSourceType || propertyName == pxr::CesiumTokens->cesiumUrl ||
        propertyName == pxr::CesiumTokens->cesiumIonAssetId ||
        propertyName == pxr::CesiumTokens->cesiumIonAccessToken ||
        propertyName == pxr::CesiumTokens->cesiumIonServerBinding ||
        propertyName == pxr::CesiumTokens->cesiumSmoothNormals ||
        propertyName == pxr::CesiumTokens->cesiumShowCreditsOnScreen ||
        propertyName == pxr::UsdTokens->material_binding) {
        // These properties trigger a reload
        tileset.value()->reload();
    }
}

void processCesiumImageryChanged(const pxr::SdfPath& imageryPath, const pxr::TfToken& propertyName) {
    const auto tilesetPath = imageryPath.GetParentPath();
    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);
    if (!tileset.has_value()) {
        return;
    }

    if (propertyName == pxr::CesiumTokens->cesiumIonAssetId ||
        propertyName == pxr::CesiumTokens->cesiumIonAccessToken ||
        propertyName == pxr::CesiumTokens->cesiumIonServerBinding ||
        propertyName == pxr::CesiumTokens->cesiumShowCreditsOnScreen) {
        // Reload the tileset that the imagery is attached to
        tileset.value()->reload();
    }

    if (propertyName == pxr::CesiumTokens->cesiumAlpha) {
        // Update the imagery layer alpha
        const auto imageryLayerIndex = tileset.value()->findImageryLayerIndex(imageryPath);
        if (imageryLayerIndex.has_value()) {
            tileset.value()->updateImageryLayerAlpha(imageryLayerIndex.value());
        }
    }
}

void processCesiumGeoreferenceChanged(const pxr::SdfPath& georeferencePath) {
    const auto globeAnchors = GlobeAnchorRegistry::getInstance().getAllAnchors();
    for (const auto& globeAnchor : globeAnchors) {

        // We only want to update an anchor if we are updating its related Georeference Prim.
        if (georeferencePath !=
            UsdUtil::getAnchorGeoreferencePath(globeAnchor->getPrimPath()).value_or(pxr::SdfPath::EmptyPath())) {
            continue;
        }

        const auto origin = UsdUtil::getCartographicOriginForAnchor(globeAnchor->getPrimPath());
        if (!origin.has_value()) {
            continue;
        }

        const auto anchorApi = UsdUtil::getCesiumGlobeAnchor(globeAnchor->getPrimPath());
        GeospatialUtil::updateAnchorOrigin(origin.value(), anchorApi, globeAnchor);
    }
}

void processCesiumGlobeAnchorChanged(const pxr::SdfPath& globeAnchorPath, const pxr::TfToken& propertyName) {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(globeAnchorPath);

    const auto origin = UsdUtil::getCartographicOriginForAnchor(globeAnchorPath);
    if (!origin.has_value()) {
        return;
    }

    bool detectTransformChanges;
    globeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    if (detectTransformChanges && (propertyName == pxr::CesiumTokens->cesiumAnchorDetectTransformChanges ||
                                   propertyName == pxr::UsdTokens->xformOp_transform_cesium)) {
        GeospatialUtil::updateAnchorByUsdTransform(origin.value(), globeAnchor);
        return;
    }

    if (propertyName == pxr::CesiumTokens->cesiumAnchorGeographicCoordinates) {
        GeospatialUtil::updateAnchorByLatLongHeight(origin.value(), globeAnchor);
        return;
    }

    if (propertyName == pxr::CesiumTokens->cesiumAnchorPosition ||
        propertyName == pxr::CesiumTokens->cesiumAnchorRotation ||
        propertyName == pxr::CesiumTokens->cesiumAnchorScale) {
        GeospatialUtil::updateAnchorByFixedTransform(origin.value(), globeAnchor);
        return;
    }
}

void processCesiumIonServerChanged(const pxr::SdfPath& ionServerPath) {
    reloadIonServerAssets(ionServerPath);
}

void processUsdShaderChanged(const pxr::SdfPath& shaderPath, const pxr::TfToken& propertyName) {
    const auto shader = UsdUtil::getUsdShader(shaderPath);
    const auto shaderPathFabric = FabricUtil::toFabricPath(shaderPath);
    const auto materialPath = shaderPath.GetParentPath();
    const auto materialPathFabric = FabricUtil::toFabricPath(materialPath);

    if (!UsdUtil::isUsdMaterial(materialPath)) {
        // Skip if parent path is not a material
        return;
    }

    const auto inputNamespace = std::string_view("inputs:");

    const auto& attributeName = propertyName.GetString();

    if (attributeName.rfind(inputNamespace) != 0) {
        // Skip if changed attribute is not a shader input
        return;
    }

    const auto inputName = pxr::TfToken(attributeName.substr(inputNamespace.size()));

    auto shaderInput = shader.GetInput(inputName);
    if (!shaderInput.IsDefined()) {
        // Skip if changed attribute is not a shader input
        return;
    }

    if (shaderInput.HasConnectedSource()) {
        // Skip if shader input is connected to something else
        return;
    }

    if (!FabricUtil::materialHasCesiumNodes(materialPathFabric)) {
        // Simple materials can be skipped. We only need to handle materials that have been copied to each tile.
        return;
    }

    if (!FabricUtil::isShaderConnectedToMaterial(materialPathFabric, shaderPathFabric)) {
        // Skip if shader is not connected to the material
        return;
    }

    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        if (tileset->getMaterialPath() == materialPath) {
            tileset->updateShaderInput(shaderPath, propertyName);
        }
    }

    FabricResourceManager::getInstance().updateShaderInput(materialPath, shaderPath, propertyName);
}

void processCesiumTilesetRemoved(const pxr::SdfPath& tilesetPath) {
    // Remove the tileset from the asset registry
    AssetRegistry::getInstance().removeTileset(tilesetPath);
}

void processCesiumImageryRemoved(const pxr::SdfPath& imageryPath) {
    // Remove the imagery from the asset registry and reload the tileset that the imagery was attached to
    const auto tilesetPath = imageryPath.GetParentPath();
    AssetRegistry::getInstance().removeImagery(imageryPath);

    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);

    if (tileset.has_value()) {
        tileset.value()->reload();
    }
}

void processCesiumGlobeAnchorRemoved(const pxr::SdfPath& globeAnchorPath) {
    const auto removed = GlobeAnchorRegistry::getInstance().removeAnchor(globeAnchorPath);

    if (!removed) {
        CESIUM_LOG_ERROR("Failed to remove anchor from registry: {}", globeAnchorPath.GetString());
    }
}

void processCesiumIonServerRemoved(const pxr::SdfPath& ionServerPath) {
    // Remove the server from the session registry
    SessionRegistry::getInstance().removeSession(ionServerPath);
    reloadIonServerAssets(ionServerPath);
}

void processCesiumTilesetAdded(const pxr::SdfPath& tilesetPath) {
    // Add the tileset to the asset registry
    const auto georeferencePath = UsdUtil::getOrCreateCesiumGeoreference().GetPath();
    AssetRegistry::getInstance().addTileset(tilesetPath, georeferencePath);
}

void processCesiumImageryAdded(const pxr::SdfPath& imageryPath) {
    // Add the imagery to the asset registry and reload the tileset that the imagery is attached to
    const auto tilesetPath = imageryPath.GetParentPath();
    AssetRegistry::getInstance().addImagery(imageryPath);

    const auto tileset = AssetRegistry::getInstance().getTilesetByPath(tilesetPath);

    if (tileset.has_value()) {
        tileset.value()->reload();
    }
}

void processCesiumGlobeAnchorAdded(const pxr::SdfPath& globeAnchorPath) {
    const auto anchorApi = UsdUtil::getCesiumGlobeAnchor(globeAnchorPath);
    const auto origin = UsdUtil::getCartographicOriginForAnchor(globeAnchorPath);
    assert(origin.has_value());
    pxr::GfVec3d coordinates;
    anchorApi.GetGeographicCoordinateAttr().Get(&coordinates);

    if (coordinates == pxr::GfVec3d(0.0, 0.0, 10.0)) {
        // Default geo coordinates. Place based on current USD position.
        GeospatialUtil::updateAnchorByUsdTransform(origin.value(), anchorApi);
    } else {
        // Provided geo coordinates. Place at correct location.
        GeospatialUtil::updateAnchorByLatLongHeight(origin.value(), anchorApi);
    }
}

void processCesiumIonServerAdded(const pxr::SdfPath& ionServerPath) {
    // Add the server to the session registry
    SessionRegistry::getInstance().addSession(
        *Context::instance().getAsyncSystem().get(), Context::instance().getHttpAssetAccessor(), ionServerPath);
    reloadIonServerAssets(ionServerPath);
}

} // namespace

void UsdNotificationHandler::onUpdateFrame() {
    for (const auto& addedPrim : _addedPrims) {
        switch (addedPrim.primType) {
            case PrimType::CESIUM_TILESET:
                processCesiumTilesetAdded(addedPrim.primPath);
                break;
            case PrimType::CESIUM_IMAGERY:
                processCesiumImageryAdded(addedPrim.primPath);
                break;
            case PrimType::CESIUM_GLOBE_ANCHOR:
                processCesiumGlobeAnchorAdded(addedPrim.primPath);
                break;
            case PrimType::CESIUM_ION_SERVER:
                processCesiumIonServerAdded(addedPrim.primPath);
                break;
            case PrimType::CESIUM_DATA:
            case PrimType::CESIUM_GEOREFERENCE:
            case PrimType::USD_SHADER:
            case PrimType::OTHER:
                break;
        }
    }

    for (const auto& removedPrim : _removedPrims) {
        switch (removedPrim.primType) {
            case PrimType::CESIUM_TILESET:
                processCesiumTilesetRemoved(removedPrim.primPath);
                break;
            case PrimType::CESIUM_IMAGERY:
                processCesiumImageryRemoved(removedPrim.primPath);
                break;
            case PrimType::CESIUM_GLOBE_ANCHOR:
                processCesiumGlobeAnchorRemoved(removedPrim.primPath);
                break;
            case PrimType::CESIUM_ION_SERVER:
                processCesiumIonServerRemoved(removedPrim.primPath);
                break;
            case PrimType::CESIUM_DATA:
            case PrimType::CESIUM_GEOREFERENCE:
            case PrimType::USD_SHADER:
            case PrimType::OTHER:
                break;
        }
    }

    for (const auto& propertyChangedPrim : _propertyChangedPrims) {
        switch (propertyChangedPrim.primType) {
            case PrimType::CESIUM_DATA:
                processCesiumDataChanged(propertyChangedPrim.propertyName);
                break;
            case PrimType::CESIUM_TILESET:
                processCesiumTilesetChanged(propertyChangedPrim.primPath, propertyChangedPrim.propertyName);
                break;
            case PrimType::CESIUM_IMAGERY:
                processCesiumImageryChanged(propertyChangedPrim.primPath, propertyChangedPrim.propertyName);
                break;
            case PrimType::CESIUM_GEOREFERENCE:
                processCesiumGeoreferenceChanged(propertyChangedPrim.primPath);
                break;
            case PrimType::CESIUM_GLOBE_ANCHOR:
                processCesiumGlobeAnchorChanged(propertyChangedPrim.primPath, propertyChangedPrim.propertyName);
                break;
            case PrimType::CESIUM_ION_SERVER:
                processCesiumIonServerChanged(propertyChangedPrim.primPath);
                break;
            case PrimType::USD_SHADER:
                processUsdShaderChanged(propertyChangedPrim.primPath, propertyChangedPrim.propertyName);
                break;
            case PrimType::OTHER:
                break;
        }
    }

    _addedPrims.clear();
    _removedPrims.clear();
    _propertyChangedPrims.clear();
}

} // namespace cesium::omniverse
