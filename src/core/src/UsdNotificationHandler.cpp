#include "cesium/omniverse/UsdNotificationHandler.h"

#include "cesium/omniverse/AssetRegistry.h"
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

ChangedPrimType getType(const pxr::SdfPath& path) {
    if (UsdUtil::primExists(path)) {
        if (UsdUtil::isCesiumData(path)) {
            return ChangedPrimType::CESIUM_DATA;
        } else if (UsdUtil::isCesiumTileset(path)) {
            return ChangedPrimType::CESIUM_TILESET;
        } else if (UsdUtil::isCesiumImagery(path)) {
            return ChangedPrimType::CESIUM_IMAGERY;
        } else if (UsdUtil::isCesiumGeoreference(path)) {
            return ChangedPrimType::CESIUM_GEOREFERENCE;
        } else if (UsdUtil::hasCesiumGlobeAnchor(path)) {
            return ChangedPrimType::CESIUM_GLOBE_ANCHOR;
        } else if (UsdUtil::isCesiumIonServer(path)) {
            return ChangedPrimType::CESIUM_ION_SERVER;
        } else if (UsdUtil::isUsdShader(path)) {
            return ChangedPrimType::USD_SHADER;
        }
    } else {
        // If the prim doesn't exist, because it was removed from the stage already, we can get the type from the asset registry
        const auto assetType = AssetRegistry::getInstance().getAssetType(path);

        switch (assetType) {
            case AssetType::TILESET:
                return ChangedPrimType::CESIUM_TILESET;
            case AssetType::IMAGERY:
                return ChangedPrimType::CESIUM_IMAGERY;
            default:
                break;
        }

        // If we still haven't found the prim type, it could be a globe anchor, and we should check if it exists in the globe anchor registry
        if (GlobeAnchorRegistry::getInstance().anchorExists(path)) {
            return ChangedPrimType::CESIUM_GLOBE_ANCHOR;
        }

        // If we still haven't found the prim type, it could be a Cesium ion session, and we should check if it exists in the session registry
        if (SessionRegistry::getInstance().sessionExists(path)) {
            return ChangedPrimType::CESIUM_ION_SERVER;
        }
    }

    return ChangedPrimType::OTHER;
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
            const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);
            if (tileset) {
                tileset->reload();
            }
        }
    }
}

void processCesiumDataChanged(const std::vector<pxr::TfToken>& properties) {
    auto reloadStage = false;

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
            property == pxr::CesiumTokens->cesiumDebugRandomColors ||
            property == pxr::CesiumTokens->cesiumDebugDisableGeoreferencing) {
            reloadStage = true;
        }
    }

    if (reloadStage) {
        Context::instance().reloadStage();
    }
}

void processCesiumTilesetChanged(const pxr::SdfPath& tilesetPath, const std::vector<pxr::TfToken>& properties) {
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);
    if (!tileset) {
        return;
    }

    auto reloadTileset = false;
    auto updateTilesetOptions = false;
    auto updateDisplayColorAndOpacity = false;

    // No change tracking needed for
    // * cesiumSuspendUpdate
    // * georeferenceBinding

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumSourceType ||
            property == pxr::CesiumTokens->cesiumUrl ||
            property == pxr::CesiumTokens->cesiumIonAssetId ||
            property == pxr::CesiumTokens->cesiumIonAccessToken ||
            property == pxr::CesiumTokens->cesiumIonServerBinding ||
            property == pxr::CesiumTokens->cesiumSmoothNormals ||
            property == pxr::CesiumTokens->cesiumShowCreditsOnScreen ||
            property == pxr::UsdTokens->material_binding) {
            reloadTileset = true;
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

    if (reloadTileset) {
        tileset->reload();
        return; // Skip other updates
    }

    if (updateTilesetOptions) {
        tileset->updateTilesetOptionsFromProperties();
    }

    if (updateDisplayColorAndOpacity) {
        tileset->updateDisplayColorAndOpacity();
    }
}

void processCesiumImageryChanged(const pxr::SdfPath& imageryPath, const std::vector<pxr::TfToken>& properties) {
    const auto tilesetPath = imageryPath.GetParentPath();
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);
    if (!tileset) {
        return;
    }

    auto reloadTileset = false;
    auto updateImageryLayerAlpha = false;

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumIonAssetId ||
            property == pxr::CesiumTokens->cesiumIonAccessToken ||
            property == pxr::CesiumTokens->cesiumIonServerBinding ||
            property == pxr::CesiumTokens->cesiumShowCreditsOnScreen) {
            reloadTileset = true;
        } else if (property == pxr::CesiumTokens->cesiumAlpha) {
            updateImageryLayerAlpha = true;
        }
    }
    // clang-format on

    if (reloadTileset) {
        tileset->reload();
        return; // Skip other updates
    }

    if (updateImageryLayerAlpha) {
        const auto imageryLayerIndex = tileset->findImageryLayerIndex(imageryPath);
        if (imageryLayerIndex.has_value()) {
            tileset->updateImageryLayerAlpha(imageryLayerIndex.value());
        }
    }
}

void processCesiumGeoreferenceChanged(
    const pxr::SdfPath& georeferencePath,
    const std::vector<pxr::TfToken>& properties) {
    auto updateGlobeAnchors = false;

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumGeoreferenceOriginLongitude ||
            property == pxr::CesiumTokens->cesiumGeoreferenceOriginLatitude ||
            property == pxr::CesiumTokens->cesiumGeoreferenceOriginHeight) {
            updateGlobeAnchors = true;
        }
    }
    // clang-format on

    if (updateGlobeAnchors) {
        const auto anchors = GlobeAnchorRegistry::getInstance().getAllAnchors();
        for (const auto& globeAnchor : anchors) {
            const auto anchorApi = UsdUtil::getCesiumGlobeAnchor(globeAnchor->getPrimPath());

            // We only want to update an anchor if we are updating its related Georeference Prim.
            if (georeferencePath !=
                UsdUtil::getAnchorGeoreferencePath(globeAnchor->getPrimPath()).value_or(pxr::SdfPath::EmptyPath())) {
                continue;
            }

            const auto origin = UsdUtil::getCartographicOriginForAnchor(globeAnchor->getPrimPath());
            if (!origin.has_value()) {
                continue;
            }

            GeospatialUtil::updateAnchorOrigin(origin.value(), anchorApi, globeAnchor);
        }
    }
}

void processCesiumGlobeAnchorChanged(const pxr::SdfPath& globeAnchorPath, const std::vector<pxr::TfToken>& properties) {
    const auto globeAnchor = UsdUtil::getCesiumGlobeAnchor(globeAnchorPath);
    const auto origin = UsdUtil::getCartographicOriginForAnchor(globeAnchorPath);

    if (!origin.has_value()) {
        return;
    }

    auto updateByUsdTransform = false;
    auto updateByLatLongHeight = false;
    auto updateByFixedTransform = false;

    bool detectTransformChanges;
    globeAnchor.GetDetectTransformChangesAttr().Get(&detectTransformChanges);

    // clang-format off
    for (const auto& property : properties) {
        if (detectTransformChanges && (property == pxr::CesiumTokens->cesiumAnchorDetectTransformChanges ||
                                       property == pxr::UsdTokens->xformOp_transform_cesium)) {
            updateByUsdTransform = true;
        } else if (property == pxr::CesiumTokens->cesiumAnchorGeographicCoordinates) {
            updateByLatLongHeight = true;
        } else if (
            property == pxr::CesiumTokens->cesiumAnchorPosition ||
            property == pxr::CesiumTokens->cesiumAnchorRotation ||
            property == pxr::CesiumTokens->cesiumAnchorScale) {
            updateByFixedTransform = true;
        }
    }
    // clang-format on

    if (updateByUsdTransform) {
        GeospatialUtil::updateAnchorByUsdTransform(origin.value(), globeAnchor);
    } else if (updateByLatLongHeight) {
        GeospatialUtil::updateAnchorByLatLongHeight(origin.value(), globeAnchor);
    } else if (updateByFixedTransform) {
        GeospatialUtil::updateAnchorByFixedTransform(origin.value(), globeAnchor);
    }
}

void processCesiumIonServerChanged(const pxr::SdfPath& ionServerPath, const std::vector<pxr::TfToken>& properties) {
    auto reloadSession = false;
    auto reloadAssets = false;

    // clang-format off
    for (const auto& property : properties) {
        if (property == pxr::CesiumTokens->cesiumIonServerUrl ||
            property == pxr::CesiumTokens->cesiumIonServerApiUrl ||
            property == pxr::CesiumTokens->cesiumIonServerApplicationId) {
            reloadSession = true;
            reloadAssets = true;
        } else if (
            property == pxr::CesiumTokens->cesiumProjectDefaultIonAccessToken ||
            property == pxr::CesiumTokens->cesiumProjectDefaultIonAccessTokenId) {
            reloadAssets = true;
        }
    }
    // clang-format on

    if (reloadSession) {
        SessionRegistry::getInstance().removeSession(ionServerPath);
        SessionRegistry::getInstance().addSession(
            *Context::instance().getAsyncSystem().get(), Context::instance().getHttpAssetAccessor(), ionServerPath);
    }

    if (reloadAssets) {
        reloadIonServerAssets(ionServerPath);
    }
}

void processUsdShaderChanged(const pxr::SdfPath& shaderPath, const std::vector<pxr::TfToken>& properties) {
    const auto shader = UsdUtil::getUsdShader(shaderPath);
    const auto shaderPathFabric = FabricUtil::toFabricPath(shaderPath);
    const auto materialPath = shaderPath.GetParentPath();
    const auto materialPathFabric = FabricUtil::toFabricPath(materialPath);

    if (!UsdUtil::isUsdMaterial(materialPath)) {
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
                tileset->updateShaderInput(shaderPath, property);
            }
        }

        FabricResourceManager::getInstance().updateShaderInput(materialPath, shaderPath, property);
    }
}

void processCesiumTilesetRemoved(const pxr::SdfPath& tilesetPath) {
    AssetRegistry::getInstance().removeTileset(tilesetPath);
}

void processCesiumImageryRemoved(const pxr::SdfPath& imageryPath) {
    AssetRegistry::getInstance().removeImagery(imageryPath);

    const auto tilesetPath = imageryPath.GetParentPath();
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);

    if (tileset) {
        tileset->reload();
    }
}

void processCesiumGlobeAnchorRemoved(const pxr::SdfPath& globeAnchorPath) {
    const auto removed = GlobeAnchorRegistry::getInstance().removeAnchor(globeAnchorPath);

    if (!removed) {
        CESIUM_LOG_ERROR("Failed to remove anchor from registry: {}", globeAnchorPath.GetString());
    }
}

void processCesiumIonServerRemoved(const pxr::SdfPath& ionServerPath) {
    SessionRegistry::getInstance().removeSession(ionServerPath);
    reloadIonServerAssets(ionServerPath);
}

void processCesiumTilesetAdded(const pxr::SdfPath& tilesetPath) {
    const auto georeferencePath = UsdUtil::getOrCreateCesiumGeoreference().GetPath();
    AssetRegistry::getInstance().addTileset(tilesetPath, georeferencePath);
}

void processCesiumImageryAdded(const pxr::SdfPath& imageryPath) {
    AssetRegistry::getInstance().addImagery(imageryPath);

    const auto tilesetPath = imageryPath.GetParentPath();
    const auto tileset = AssetRegistry::getInstance().getTileset(tilesetPath);

    if (tileset) {
        tileset->reload();
    }
}

void processCesiumGlobeAnchorAdded(const pxr::SdfPath& globeAnchorPath) {
    const auto anchorApi = UsdUtil::getCesiumGlobeAnchor(globeAnchorPath);
    const auto origin = UsdUtil::getCartographicOriginForAnchor(globeAnchorPath);
    assert(origin.has_value());
    pxr::GfVec3d coordinates;
    anchorApi.GetGeographicCoordinateAttr().Get(&coordinates);

    if (coordinates == pxr::GfVec3d{0.0, 0.0, 10.0}) {
        // Default geo coordinates. Place based on current USD position.
        GeospatialUtil::updateAnchorByUsdTransform(origin.value(), anchorApi);
    } else {
        // Provided geo coordinates. Place at correct location.
        GeospatialUtil::updateAnchorByLatLongHeight(origin.value(), anchorApi);
    }
}

void processCesiumIonServerAdded(const pxr::SdfPath& ionServerPath) {
    SessionRegistry::getInstance().addSession(
        *Context::instance().getAsyncSystem().get(), Context::instance().getHttpAssetAccessor(), ionServerPath);
    reloadIonServerAssets(ionServerPath);
}

std::vector<ChangedPrim> consolidateChangedPrims(const std::vector<ChangedPrim>& changedPrims) {
    // Consolidate changes while preserving original insertion order
    std::vector<ChangedPrim> consolidated;

    ChangedPrim* previous = nullptr;

    for (const auto& changedPrim : changedPrims) {
        if (previous != nullptr && changedPrim.primPath == previous->primPath) {
            if (previous->changeType == ChangeType::PRIM_ADDED &&
                changedPrim.changeType == ChangeType::PROPERTY_CHANGED) {
                // Ignore property changes that occur immediately after the prim is added. This avoids unecessary churn.
                continue;
            }

            if (previous->changeType == ChangeType::PROPERTY_CHANGED &&
                changedPrim.changeType == ChangeType::PROPERTY_CHANGED) {
                // Consolidate property changes so that they can be processed together
                previous->properties.insert(
                    previous->properties.end(), changedPrim.properties.begin(), changedPrim.properties.end());
                continue;
            }
        }

        consolidated.push_back(changedPrim);

        previous = &consolidated.back();
    }

    return consolidated;
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

    // We need to add prims manually because USD doesn't notify us about resynced paths when the stage is loaded
    for (const auto& prim : stage->Traverse()) {
        const auto type = getType(prim.GetPath());
        if (type != ChangedPrimType::OTHER) {
            insertAddedPrim(prim.GetPath(), type);
        }
    }

    // Process changes immediately
    onUpdateFrame();
}

void UsdNotificationHandler::onUpdateFrame() {
    const auto changedPrims = consolidateChangedPrims(_changedPrims);

    // Reset for next frame
    _changedPrims.clear();

    for (const auto& changedPrim : changedPrims) {
        switch (changedPrim.changeType) {
            case ChangeType::PROPERTY_CHANGED:
                switch (changedPrim.primType) {
                    case ChangedPrimType::CESIUM_DATA:
                        processCesiumDataChanged(changedPrim.properties);
                        break;
                    case ChangedPrimType::CESIUM_TILESET:
                        processCesiumTilesetChanged(changedPrim.primPath, changedPrim.properties);
                        break;
                    case ChangedPrimType::CESIUM_IMAGERY:
                        processCesiumImageryChanged(changedPrim.primPath, changedPrim.properties);
                        break;
                    case ChangedPrimType::CESIUM_GEOREFERENCE:
                        processCesiumGeoreferenceChanged(changedPrim.primPath, changedPrim.properties);
                        break;
                    case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                        processCesiumGlobeAnchorChanged(changedPrim.primPath, changedPrim.properties);
                        break;
                    case ChangedPrimType::CESIUM_ION_SERVER:
                        processCesiumIonServerChanged(changedPrim.primPath, changedPrim.properties);
                        break;
                    case ChangedPrimType::USD_SHADER:
                        processUsdShaderChanged(changedPrim.primPath, changedPrim.properties);
                        break;
                    case ChangedPrimType::OTHER:
                        break;
                }
                break;
            case ChangeType::PRIM_ADDED:
                switch (changedPrim.primType) {
                    case ChangedPrimType::CESIUM_TILESET:
                        processCesiumTilesetAdded(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_IMAGERY:
                        processCesiumImageryAdded(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                        processCesiumGlobeAnchorAdded(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_ION_SERVER:
                        processCesiumIonServerAdded(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_DATA:
                    case ChangedPrimType::CESIUM_GEOREFERENCE:
                    case ChangedPrimType::USD_SHADER:
                    case ChangedPrimType::OTHER:
                        break;
                }
                break;
            case ChangeType::PRIM_REMOVED:
                switch (changedPrim.primType) {
                    case ChangedPrimType::CESIUM_TILESET:
                        processCesiumTilesetRemoved(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_IMAGERY:
                        processCesiumImageryRemoved(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_GLOBE_ANCHOR:
                        processCesiumGlobeAnchorRemoved(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_ION_SERVER:
                        processCesiumIonServerRemoved(changedPrim.primPath);
                        break;
                    case ChangedPrimType::CESIUM_DATA:
                    case ChangedPrimType::CESIUM_GEOREFERENCE:
                    case ChangedPrimType::USD_SHADER:
                    case ChangedPrimType::OTHER:
                        break;
                }
                break;
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
                const auto isTileset = getType(path) == ChangedPrimType::CESIUM_TILESET;
                const auto isTilesetAlreadyRegistered = AssetRegistry::getInstance().getTileset(path) != nullptr;

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
    if (type != ChangedPrimType::OTHER) {
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
    // loop through items in the asset registry.
    const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();
    for (const auto& tileset : tilesets) {
        const auto tilesetPath = tileset->getPath();
        const auto type = getType(tilesetPath);
        if (type == ChangedPrimType::CESIUM_TILESET) {
            if (inSubtree(primPath, tilesetPath)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }

    const auto& imageries = AssetRegistry::getInstance().getAllImageries();
    for (const auto& imagery : imageries) {
        const auto imageryPath = imagery->getPath();
        const auto type = getType(imageryPath);
        if (type == ChangedPrimType::CESIUM_IMAGERY) {
            if (inSubtree(primPath, imageryPath)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }

    const auto& anchors = GlobeAnchorRegistry::getInstance().getAllAnchorPaths();
    for (const auto& anchorPath : anchors) {
        const auto& path = pxr::SdfPath(anchorPath);
        const auto& type = getType(path);
        if (type == ChangedPrimType::CESIUM_GLOBE_ANCHOR) {
            if (inSubtree(primPath, path)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }

    const auto& servers = SessionRegistry::getInstance().getAllServerPaths();
    for (const auto& path : servers) {
        const auto& type = getType(path);
        if (type == ChangedPrimType::CESIUM_ION_SERVER) {
            if (inSubtree(primPath, path)) {
                insertRemovedPrim(primPath, type);
            }
        }
    }
}

void UsdNotificationHandler::onPropertyChanged(const pxr::SdfPath& propertyPath) {
    const auto& propertyName = propertyPath.GetNameToken();
    const auto primPath = propertyPath.GetPrimPath();
    const auto type = getType(primPath);
    if (type != ChangedPrimType::OTHER) {
        insertPropertyChanged(primPath, type, propertyName);
    }
}

void UsdNotificationHandler::insertAddedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType) {
    // In C++ 20 the ChangePrim{} wrapper can be removed
    _changedPrims.emplace_back(ChangedPrim{primPath, {}, primType, ChangeType::PRIM_ADDED});
    CESIUM_LOG_TRACE("Added prim: {}", primPath.GetText());
}

void UsdNotificationHandler::insertRemovedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType) {
    // In C++ 20 the ChangePrim{} wrapper can be removed
    _changedPrims.emplace_back(ChangedPrim{primPath, {}, primType, ChangeType::PRIM_REMOVED});
    CESIUM_LOG_TRACE("Removed prim: {}", primPath.GetText());
}

void UsdNotificationHandler::insertPropertyChanged(
    const pxr::SdfPath& primPath,
    ChangedPrimType primType,
    const pxr::TfToken& propertyName) {
    // In C++ 20 the ChangePrim{} wrapper can be removed
    _changedPrims.emplace_back(ChangedPrim{primPath, {propertyName}, primType, ChangeType::PROPERTY_CHANGED});
    CESIUM_LOG_TRACE("Property changed: {} {}", primPath.GetText(), propertyName.GetText());
}

} // namespace cesium::omniverse
