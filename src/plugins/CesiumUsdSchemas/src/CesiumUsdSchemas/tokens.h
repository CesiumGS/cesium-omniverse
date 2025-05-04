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
    /// \brief "cesium:east"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumEast;
    /// \brief "cesium:ecefToUsdTransform"
    /// 
    /// CesiumGeoreference
    const TfToken cesiumEcefToUsdTransform;
    /// \brief "cesium:ellipsoidBinding"
    /// 
    /// CesiumGeoreference
    const TfToken cesiumEllipsoidBinding;
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
    /// \brief "cesium:format"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumFormat;
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
    /// CesiumTileset, CesiumIonRasterOverlay
    const TfToken cesiumIonAccessToken;
    /// \brief "cesium:ionAssetId"
    /// 
    /// CesiumTileset, CesiumIonRasterOverlay
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
    /// CesiumTileset, CesiumIonRasterOverlay
    const TfToken cesiumIonServerBinding;
    /// \brief "cesium:ionServerUrl"
    /// 
    /// CesiumIonServer
    const TfToken cesiumIonServerUrl;
    /// \brief "cesium:layer"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumLayer;
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
    /// CesiumTileset, CesiumRasterOverlay
    const TfToken cesiumMaximumScreenSpaceError;
    /// \brief "cesium:maximumSimultaneousTileLoads"
    /// 
    /// CesiumTileset, CesiumRasterOverlay
    const TfToken cesiumMaximumSimultaneousTileLoads;
    /// \brief "cesium:maximumTextureSize"
    /// 
    /// CesiumRasterOverlay
    const TfToken cesiumMaximumTextureSize;
    /// \brief "cesium:maximumZoomLevel"
    /// 
    /// CesiumTileMapServiceRasterOverlay, CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumMaximumZoomLevel;
    /// \brief "cesium:minimumLevel"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumMinimumLevel;
    /// \brief "cesium:minimumZoomLevel"
    /// 
    /// CesiumTileMapServiceRasterOverlay, CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumMinimumZoomLevel;
    /// \brief "cesium:north"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumNorth;
    /// \brief "cesium:overlayRenderMethod"
    /// 
    /// CesiumRasterOverlay, CesiumPolygonRasterOverlay
    const TfToken cesiumOverlayRenderMethod;
    /// \brief "cesium:pointSize"
    /// 
    /// CesiumTileset
    const TfToken cesiumPointSize;
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
    /// \brief "cesium:radii"
    /// 
    /// CesiumEllipsoid
    const TfToken cesiumRadii;
    /// \brief "cesium:rasterOverlayBinding"
    /// 
    /// CesiumTileset
    const TfToken cesiumRasterOverlayBinding;
    /// \brief "cesium:rootTilesX"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumRootTilesX;
    /// \brief "cesium:rootTilesY"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumRootTilesY;
    /// \brief "cesium:selectedIonServer"
    /// 
    /// CesiumData
    const TfToken cesiumSelectedIonServer;
    /// \brief "cesium:showCreditsOnScreen"
    /// 
    /// CesiumTileset, CesiumRasterOverlay
    const TfToken cesiumShowCreditsOnScreen;
    /// \brief "cesium:smoothNormals"
    /// 
    /// CesiumTileset
    const TfToken cesiumSmoothNormals;
    /// \brief "cesium:sourceType"
    /// 
    /// CesiumTileset
    const TfToken cesiumSourceType;
    /// \brief "cesium:south"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumSouth;
    /// \brief "cesium:specifyTileMatrixSetLabels"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumSpecifyTileMatrixSetLabels;
    /// \brief "cesium:specifyTilingScheme"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumSpecifyTilingScheme;
    /// \brief "cesium:specifyZoomLevels"
    /// 
    /// CesiumTileMapServiceRasterOverlay, CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumSpecifyZoomLevels;
    /// \brief "cesium:style"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumStyle;
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
    /// \brief "cesium:tileMatrixSetId"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumTileMatrixSetId;
    /// \brief "cesium:tileMatrixSetLabelPrefix"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumTileMatrixSetLabelPrefix;
    /// \brief "cesium:tileMatrixSetLabels"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumTileMatrixSetLabels;
    /// \brief "cesium:tileWidth"
    /// 
    /// CesiumWebMapServiceRasterOverlay
    const TfToken cesiumTileWidth;
    /// \brief "cesium:url"
    /// 
    /// CesiumTileset, CesiumTileMapServiceRasterOverlay, CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumUrl;
    /// \brief "cesium:useWebMercatorProjection"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumUseWebMercatorProjection;
    /// \brief "cesium:west"
    /// 
    /// CesiumWebMapTileServiceRasterOverlay
    const TfToken cesiumWest;
    /// \brief "clip"
    /// 
    /// Possible value for CesiumRasterOverlay::GetOverlayRenderMethodAttr(), Fallback value for CesiumPolygonRasterOverlay::GetCesiumOverlayRenderMethodAttr()
    const TfToken clip;
    /// \brief "ion"
    /// 
    /// Fallback value for CesiumTileset::GetSourceTypeAttr()
    const TfToken ion;
    /// \brief "overlay"
    /// 
    /// Fallback value for CesiumRasterOverlay::GetOverlayRenderMethodAttr()
    const TfToken overlay;
    /// \brief "url"
    /// 
    /// Possible value for CesiumTileset::GetSourceTypeAttr()
    const TfToken url;
    /// \brief "CesiumDataPrim"
    /// 
    /// Schema identifer and family for CesiumData
    const TfToken CesiumDataPrim;
    /// \brief "CesiumEllipsoidPrim"
    /// 
    /// Schema identifer and family for CesiumEllipsoid
    const TfToken CesiumEllipsoidPrim;
    /// \brief "CesiumGeoreferencePrim"
    /// 
    /// Schema identifer and family for CesiumGeoreference
    const TfToken CesiumGeoreferencePrim;
    /// \brief "CesiumGlobeAnchorSchemaAPI"
    /// 
    /// Schema identifer and family for CesiumGlobeAnchorAPI
    const TfToken CesiumGlobeAnchorSchemaAPI;
    /// \brief "CesiumIonRasterOverlayPrim"
    /// 
    /// Schema identifer and family for CesiumIonRasterOverlay
    const TfToken CesiumIonRasterOverlayPrim;
    /// \brief "CesiumIonServerPrim"
    /// 
    /// Schema identifer and family for CesiumIonServer
    const TfToken CesiumIonServerPrim;
    /// \brief "CesiumPolygonRasterOverlayPrim"
    /// 
    /// Schema identifer and family for CesiumPolygonRasterOverlay
    const TfToken CesiumPolygonRasterOverlayPrim;
    /// \brief "CesiumRasterOverlayPrim"
    /// 
    /// Schema identifer and family for CesiumRasterOverlay
    const TfToken CesiumRasterOverlayPrim;
    /// \brief "CesiumTileMapServiceRasterOverlayPrim"
    /// 
    /// Schema identifer and family for CesiumTileMapServiceRasterOverlay
    const TfToken CesiumTileMapServiceRasterOverlayPrim;
    /// \brief "CesiumTilesetPrim"
    /// 
    /// Schema identifer and family for CesiumTileset
    const TfToken CesiumTilesetPrim;
    /// \brief "CesiumWebMapServiceRasterOverlayPrim"
    /// 
    /// Schema identifer and family for CesiumWebMapServiceRasterOverlay
    const TfToken CesiumWebMapServiceRasterOverlayPrim;
    /// \brief "CesiumWebMapTileServiceRasterOverlayPrim"
    /// 
    /// Schema identifer and family for CesiumWebMapTileServiceRasterOverlay
    const TfToken CesiumWebMapTileServiceRasterOverlayPrim;
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
