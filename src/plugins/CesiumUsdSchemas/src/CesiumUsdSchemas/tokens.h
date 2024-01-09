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
    /// CesiumImagery
    const TfToken cesiumAlpha;
    /// \brief "cesium:anchor:adjustOrientationForGlobeWhenMoving"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorAdjustOrientationForGlobeWhenMoving;
    /// \brief "cesium:anchor:detectTransformChanges"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorDetectTransformChanges;
    /// \brief "cesium:anchor:geographicCoordinates"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorGeographicCoordinates;
    /// \brief "cesium:anchor:georeferenceBinding"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorGeoreferenceBinding;
    /// \brief "cesium:anchor:position"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorPosition;
    /// \brief "cesium:anchor:rotation"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorRotation;
    /// \brief "cesium:anchor:scale"
    /// 
    /// CesiumGlobeAnchorAPI
    const TfToken cesiumAnchorScale;
    /// \brief "cesium:cartographicPolygonBinding"
    /// 
    /// CesiumPolygonImagery
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
    /// \brief "cesium:ionAccessToken"
    /// 
    /// CesiumIonImagery, CesiumTileset
    const TfToken cesiumIonAccessToken;
    /// \brief "cesium:ionAssetId"
    /// 
    /// CesiumIonImagery, CesiumTileset
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
    /// CesiumIonImagery, CesiumTileset
    const TfToken cesiumIonServerBinding;
    /// \brief "cesium:ionServerUrl"
    /// 
    /// CesiumIonServer
    const TfToken cesiumIonServerUrl;
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
    /// \brief "cesium:maximumScreenSpaceError"
    /// 
    /// CesiumTileset
    const TfToken cesiumMaximumScreenSpaceError;
    /// \brief "cesium:maximumSimultaneousTileLoads"
    /// 
    /// CesiumTileset
    const TfToken cesiumMaximumSimultaneousTileLoads;
    /// \brief "cesium:overlayRenderPipe"
    /// 
    /// CesiumImagery
    const TfToken cesiumOverlayRenderPipe;
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
    /// CesiumIonServer, CesiumData
    const TfToken cesiumProjectDefaultIonAccessToken;
    /// \brief "cesium:projectDefaultIonAccessTokenId"
    /// 
    /// CesiumIonServer, CesiumData
    const TfToken cesiumProjectDefaultIonAccessTokenId;
    /// \brief "cesium:selectedIonServer"
    /// 
    /// CesiumData
    const TfToken cesiumSelectedIonServer;
    /// \brief "cesium:showCreditsOnScreen"
    /// 
    /// CesiumImagery, CesiumTileset
    const TfToken cesiumShowCreditsOnScreen;
    /// \brief "cesium:smoothNormals"
    /// 
    /// CesiumTileset
    const TfToken cesiumSmoothNormals;
    /// \brief "cesium:sourceType"
    /// 
    /// CesiumTileset
    const TfToken cesiumSourceType;
    /// \brief "cesium:suspendUpdate"
    /// 
    /// CesiumTileset
    const TfToken cesiumSuspendUpdate;
    /// \brief "cesium:url"
    /// 
    /// CesiumTileset
    const TfToken cesiumUrl;
    /// \brief "clip"
    /// 
    /// Possible value for CesiumImagery::GetOverlayRenderPipeAttr()
    const TfToken clip;
    /// \brief "ion"
    /// 
    /// Possible value for CesiumTileset::GetSourceTypeAttr(), Default value for CesiumTileset::GetSourceTypeAttr()
    const TfToken ion;
    /// \brief "overlay"
    /// 
    /// Possible value for CesiumImagery::GetOverlayRenderPipeAttr(), Default value for CesiumImagery::GetOverlayRenderPipeAttr()
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
