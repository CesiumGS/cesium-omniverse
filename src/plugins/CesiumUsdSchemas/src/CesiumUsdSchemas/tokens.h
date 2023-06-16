#ifndef CESIUM_TOKENS_H
#define CESIUM_TOKENS_H

/// \file cesium/tokens.h

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
///     gprim.GetMyTokenValuedAttr().Set(CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving);
/// \endcode
struct CesiumTokensType {
    CESIUM_API CesiumTokensType();
    /// \brief "cesium:anchor:adjustOrientationForGlobeWhenMoving"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorAdjustOrientationForGlobeWhenMoving;
    /// \brief "cesium:anchor:detectTransformChanges"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorDetectTransformChanges;
    /// \brief "cesium:anchor:height"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorHeight;
    /// \brief "cesium:anchor:latitude"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorLatitude;
    /// \brief "cesium:anchor:longitude"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorLongitude;
    /// \brief "cesium:anchor:position"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorPosition;
    /// \brief "cesium:anchor:rotation"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorRotation;
    /// \brief "cesium:anchor:scale"
    /// 
    /// CesiumGlobalAnchorAPI
    const TfToken cesiumAnchorScale;
    /// \brief "cesium:culledScreenSpaceError"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumCulledScreenSpaceError;
    /// \brief "cesium:debug:disableGeometryPool"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableGeometryPool;
    /// \brief "cesium:debug:disableMaterialPool"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableMaterialPool;
    /// \brief "cesium:debug:disableMaterials"
    /// 
    /// CesiumData
    const TfToken cesiumDebugDisableMaterials;
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
    /// \brief "cesium:ecefToUsdTransform"
    /// 
    /// CesiumSession
    const TfToken cesiumEcefToUsdTransform;
    /// \brief "cesium:enableFogCulling"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumEnableFogCulling;
    /// \brief "cesium:enableFrustumCulling"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumEnableFrustumCulling;
    /// \brief "cesium:enforceCulledScreenSpaceError"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumEnforceCulledScreenSpaceError;
    /// \brief "cesium:forbidHoles"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumForbidHoles;
    /// \brief "cesium:georeferenceBinding"
    /// 
    /// CesiumTilesetAPI
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
    /// CesiumImagery, CesiumTilesetAPI
    const TfToken cesiumIonAccessToken;
    /// \brief "cesium:ionAssetId"
    /// 
    /// CesiumImagery, CesiumTilesetAPI
    const TfToken cesiumIonAssetId;
    /// \brief "cesium:loadingDescendantLimit"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumLoadingDescendantLimit;
    /// \brief "cesium:maximumCachedBytes"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumMaximumCachedBytes;
    /// \brief "cesium:maximumScreenSpaceError"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumMaximumScreenSpaceError;
    /// \brief "cesium:maximumSimultaneousTileLoads"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumMaximumSimultaneousTileLoads;
    /// \brief "cesium:preloadAncestors"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumPreloadAncestors;
    /// \brief "cesium:preloadSiblings"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumPreloadSiblings;
    /// \brief "cesium:projectDefaultIonAccessToken"
    /// 
    /// CesiumData
    const TfToken cesiumProjectDefaultIonAccessToken;
    /// \brief "cesium:projectDefaultIonAccessTokenId"
    /// 
    /// CesiumData
    const TfToken cesiumProjectDefaultIonAccessTokenId;
    /// \brief "cesium:showCreditsOnScreen"
    /// 
    /// CesiumImagery, CesiumTilesetAPI
    const TfToken cesiumShowCreditsOnScreen;
    /// \brief "cesium:smoothNormals"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumSmoothNormals;
    /// \brief "cesium:sourceType"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumSourceType;
    /// \brief "cesium:suspendUpdate"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumSuspendUpdate;
    /// \brief "cesium:url"
    /// 
    /// CesiumTilesetAPI
    const TfToken cesiumUrl;
    /// \brief "ion"
    /// 
    /// Possible value for CesiumTilesetAPI::GetCesiumSourceTypeAttr(), Default value for CesiumTilesetAPI::GetCesiumSourceTypeAttr()
    const TfToken ion;
    /// \brief "url"
    /// 
    /// Possible value for CesiumTilesetAPI::GetCesiumSourceTypeAttr()
    const TfToken url;
    /// A vector of all of the tokens listed above.
    const std::vector<TfToken> allTokens;
};

/// \var CesiumTokens
///
/// A global variable with static, efficient \link TfToken TfTokens\endlink
/// for use in all public USD API.  \sa CesiumTokensType
extern CESIUM_API TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE

#endif
