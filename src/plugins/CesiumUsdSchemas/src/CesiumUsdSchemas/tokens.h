#ifndef CESIUM_TOKENS_H
#define CESIUM_TOKENS_H

/// \file CesiumUsdSchemas/tokens.h

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// 
// This is an automatically generated file (by usdGenSchema.py).
// Do not hand-edit!
// 
// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#include "pxr/pxr.h"
#include ".//api.h"
#include "pxr/base/tf/staticData.h"
#include "pxr/base/tf/token.h"
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE


/// \class CesiumTokensType
///
/// \link CesiumTokens \endlink provides static, efficient
/// \link TfToken TfTokens\endlink for use in all public USD API.
///
/// These tokens are auto-generated from the module's schema, representing
/// property names, for when you need to fetch an attribute or relationship
/// directly by name, e.g. UsdPrim::GetAttribute(), in the most efficient
/// manner, and allow the compiler to verify that you spelled the name
/// correctly.
///
/// CesiumTokens also contains all of the \em allowedTokens values
/// declared for schema builtin attributes of 'token' scene description type.
/// Use CesiumTokens like so:
///
/// \code
///     gprim.GetMyTokenValuedAttr().Set(CesiumTokens->cesiumAlpha);
/// \endcode
struct CesiumTokensType {
    CESIUMUSDSCHEMAS_API CesiumTokensType();
    /// \brief "cesium:alpha"
    /// 
    /// CesiumRasterOverlay
    const TfToken cesiumAlpha;
    /// \brief "cesium:anchor:adjustOrientationForGlobeWhenMoving"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorAdjustOrientationForGlobeWhenMoving;
    /// \brief "cesium:anchor:detectTransformChanges"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorDetectTransformChanges;
    /// \brief "cesium:anchor:georeferenceBinding"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorGeoreferenceBinding;
    /// \brief "cesium:anchor:height"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorHeight;
    /// \brief "cesium:anchor:latitude"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorLatitude;
    /// \brief "cesium:anchor:longitude"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorLongitude;
    /// \brief "cesium:anchor:position"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorPosition;
    /// \brief "cesium:baseUrl"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumBaseUrl;
    /// \brief "cesium:cartographicPolygonBinding"
    /// 
    /// CesiumPolygonRasterOverlay
    const TfToken cesiumCartographicPolygonBinding;
    /// \brief "cesium:culledScreenSpaceError"
    /// 
    /// CesiumTileset
    const TfToken cesiumCulledScreenSpaceError;
    /// \brief "cesium:debug:disableGeometryPool"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableGeometryPool;
    /// \brief "cesium:debug:disableGeoreferencing"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableGeoreferencing;
    /// \brief "cesium:debug:disableMaterialPool"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableMaterialPool;
    /// \brief "cesium:debug:disableMaterials"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableMaterials;
    /// \brief "cesium:debug:disableTexturePool"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableTexturePool;
    /// \brief "cesium:debug:disableTextures"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableTextures;
    /// \brief "cesium:debug:geometryPoolInitialCapacity"
    /// 
    /// CesiumData
    const TfToken cesiumDebugGeometryPoolInitialCapacity;
    /// \brief "cesium:debug:materialPoolInitialCapacity"
    /// 
    /// CesiumData
    const TfToken cesiumDebugMaterialPoolInitialCapacity;
    /// \brief "cesium:debug:randomColors"
    /// 
    /// CesiumData
    const TfToken cesiumDebugRandomColors;
    /// \brief "cesium:debug:texturePoolInitialCapacity"
    /// 
    /// CesiumData
    const TfToken cesiumDebugTexturePoolInitialCapacity;
    /// \brief "cesium:displayName"
    /// 
    /// CesiumIonServer
    const TfToken cesiumDisplayName;
    /// \brief "cesium:ecefToUsdTransform"
    /// 
    /// CesiumSession
    const TfToken cesiumEcefToUsdTransform;
    /// \brief "cesium:enableFogCulling"
    /// 
    /// CesiumTileset
    const TfToken cesiumEnableFogCulling;
    /// \brief "cesium:enableFrustumCulling"
    /// 
    /// CesiumTileset
    const TfToken cesiumEnableFrustumCulling;
    /// \brief "cesium:enforceCulledScreenSpaceError"
    /// 
    /// CesiumTileset
    const TfToken cesiumEnforceCulledScreenSpaceError;
    /// \brief "cesium:excludeSelectedTiles"
    /// 
    /// CesiumPolygonRasterOverlay
    const TfToken cesiumExcludeSelectedTiles;
    /// \brief "cesium:forbidHoles"
    /// 
    /// CesiumTileset
    const TfToken cesiumForbidHoles;
    /// \brief "cesium:georeferenceBinding"
    /// 
    /// CesiumTileset
    const TfToken cesiumGeoreferenceBinding;
    /// \brief "cesium:georeferenceOrigin:height"
    /// 
    /// CesiumGeoreference
    const TfToken cesiumGeoreferenceOriginHeight;
    /// \brief "cesium:georeferenceOrigin:latitude"
    /// 
    /// CesiumGeoreference
    const TfToken cesiumGeoreferenceOriginLatitude;
    /// \brief "cesium:georeferenceOrigin:longitude"
    /// 
    /// CesiumGeoreference
    const TfToken cesiumGeoreferenceOriginLongitude;
    /// \brief "cesium:invertSelection"
    /// 
    /// CesiumPolygonRasterOverlay
    const TfToken cesiumInvertSelection;
    /// \brief "cesium:ionAccessToken"
    /// 
    /// CesiumIonRasterOverlay, CesiumTileset
    const TfToken cesiumIonAccessToken;
    /// \brief "cesium:ionAssetId"
    /// 
    /// CesiumIonRasterOverlay, CesiumTileset
    const TfToken cesiumIonAssetId;
    /// \brief "cesium:ionServerApiUrl"
    /// 
    /// CesiumIonServer
    const TfToken cesiumIonServerApiUrl;
    /// \brief "cesium:ionServerApplicationId"
    /// 
    /// CesiumIonServer
    const TfToken cesiumIonServerApplicationId;
    /// \brief "cesium:ionServerBinding"
    /// 
    /// CesiumIonRasterOverlay, CesiumTileset
    const TfToken cesiumIonServerBinding;
    /// \brief "cesium:ionServerUrl"
    /// 
    /// CesiumIonServer
    const TfToken cesiumIonServerUrl;
    /// \brief "cesium:layers"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumLayers;
    /// \brief "cesium:loadingDescendantLimit"
    /// 
    /// CesiumTileset
    const TfToken cesiumLoadingDescendantLimit;
    /// \brief "cesium:mainThreadLoadingTimeLimit"
    /// 
    /// CesiumTileset
    const TfToken cesiumMainThreadLoadingTimeLimit;
    /// \brief "cesium:maximumCachedBytes"
    /// 
    /// CesiumTileset
    const TfToken cesiumMaximumCachedBytes;
    /// \brief "cesium:maximumLevel"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumMaximumLevel;
    /// \brief "cesium:maximumScreenSpaceError"
    /// 
    /// CesiumRasterOverlay, CesiumTileset
    const TfToken cesiumMaximumScreenSpaceError;
    /// \brief "cesium:maximumSimultaneousTileLoads"
    /// 
    /// CesiumRasterOverlay, CesiumTileset
    const TfToken cesiumMaximumSimultaneousTileLoads;
    /// \brief "cesium:maximumTextureSize"
    /// 
    /// CesiumRasterOverlay
    const TfToken cesiumMaximumTextureSize;
    /// \brief "cesium:minimumLevel"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumMinimumLevel;
    /// \brief "cesium:overlayRenderMethod"
    /// 
    /// CesiumPolygonRasterOverlay, CesiumRasterOverlay
    const TfToken cesiumOverlayRenderMethod;
    /// \brief "cesium:preloadAncestors"
    /// 
    /// CesiumTileset
    const TfToken cesiumPreloadAncestors;
    /// \brief "cesium:preloadSiblings"
    /// 
    /// CesiumTileset
    const TfToken cesiumPreloadSiblings;
    /// \brief "cesium:projectDefaultIonAccessToken"
    /// 
    /// CesiumIonServer
    const TfToken cesiumProjectDefaultIonAccessToken;
    /// \brief "cesium:projectDefaultIonAccessTokenId"
    /// 
    /// CesiumIonServer
    const TfToken cesiumProjectDefaultIonAccessTokenId;
    /// \brief "cesium:rasterOverlayBinding"
    /// 
    /// CesiumTileset
    const TfToken cesiumRasterOverlayBinding;
    /// \brief "cesium:selectedIonServer"
    /// 
    /// CesiumData
    const TfToken cesiumSelectedIonServer;
    /// \brief "cesium:showCreditsOnScreen"
    /// 
    /// CesiumRasterOverlay, CesiumTileset
    const TfToken cesiumShowCreditsOnScreen;
    /// \brief "cesium:smoothNormals"
    /// 
    /// CesiumTileset
    const TfToken cesiumSmoothNormals;
    /// \brief "cesium:sourceType"
    /// 
    /// CesiumTileset
    const TfToken cesiumSourceType;
    /// \brief "cesium:subTileCacheBytes"
    /// 
    /// CesiumRasterOverlay
    const TfToken cesiumSubTileCacheBytes;
    /// \brief "cesium:suspendUpdate"
    /// 
    /// CesiumTileset
    const TfToken cesiumSuspendUpdate;
    /// \brief "cesium:tileHeight"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumTileHeight;
    /// \brief "cesium:tileWidth"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumTileWidth;
    /// \brief "cesium:url"
    /// 
    /// CesiumTileset
    const TfToken cesiumUrl;
    /// \brief "clip"
    /// 
    /// Default value for CesiumPolygonRasterOverlay::GetCesiumOverlayRenderMethodAttr(), Possible value for CesiumRasterOverlay::GetOverlayRenderMethodAttr()
    const TfToken clip;
    /// \brief "ion"
    /// 
    /// Possible value for CesiumTileset::GetSourceTypeAttr(), Default value for CesiumTileset::GetSourceTypeAttr()
    const TfToken ion;
    /// \brief "overlay"
    /// 
    /// Possible value for CesiumRasterOverlay::GetOverlayRenderMethodAttr(), Default value for CesiumRasterOverlay::GetOverlayRenderMethodAttr()
    const TfToken overlay;
    /// \brief "url"
    /// 
    /// Possible value for CesiumTileset::GetSourceTypeAttr()
    const TfToken url;
    /// A vector of all of the tokens listed above.
    const std::vector<TfToken> allTokens;
};

/// \var CesiumTokens
///
/// A global variable with static, efficient \link TfToken TfTokens\endlink
/// for use in all public USD API.  \sa CesiumTokensType
extern CESIUMUSDSCHEMAS_API TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE

#endif
