#include ".//tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    cesiumCulledScreenSpaceError("cesium:culledScreenSpaceError", TfToken::Immortal),
    cesiumEnableFogCulling("cesium:enableFogCulling", TfToken::Immortal),
    cesiumEnableFrustumCulling("cesium:enableFrustumCulling", TfToken::Immortal),
    cesiumEnforceCulledScreenSpaceError("cesium:enforceCulledScreenSpaceError", TfToken::Immortal),
    cesiumForbidHoles("cesium:forbidHoles", TfToken::Immortal),
    cesiumGeoreferenceOriginHeight("cesium:georeferenceOrigin:height", TfToken::Immortal),
    cesiumGeoreferenceOriginLatitude("cesium:georeferenceOrigin:latitude", TfToken::Immortal),
    cesiumGeoreferenceOriginLongitude("cesium:georeferenceOrigin:longitude", TfToken::Immortal),
    cesiumIonAccessToken("cesium:ionAccessToken", TfToken::Immortal),
    cesiumIonAssetId("cesium:ionAssetId", TfToken::Immortal),
    cesiumLoadingDescendantLimit("cesium:loadingDescendantLimit", TfToken::Immortal),
    cesiumMaximumCachedBytes("cesium:maximumCachedBytes", TfToken::Immortal),
    cesiumMaximumScreenSpaceError("cesium:maximumScreenSpaceError", TfToken::Immortal),
    cesiumMaximumSimultaneousTileLoads("cesium:maximumSimultaneousTileLoads", TfToken::Immortal),
    cesiumPreloadAncestors("cesium:preloadAncestors", TfToken::Immortal),
    cesiumPreloadSiblings("cesium:preloadSiblings", TfToken::Immortal),
    cesiumProjectDefaultIonAccessToken("cesium:projectDefaultIonAccessToken", TfToken::Immortal),
    cesiumProjectDefaultIonAccessTokenId("cesium:projectDefaultIonAccessTokenId", TfToken::Immortal),
    cesiumSmoothNormals("cesium:smoothNormals", TfToken::Immortal),
    cesiumSourceType("cesium:sourceType", TfToken::Immortal),
    cesiumSuspendUpdate("cesium:suspendUpdate", TfToken::Immortal),
    cesiumUrl("cesium:url", TfToken::Immortal),
    ion("ion", TfToken::Immortal),
    url("url", TfToken::Immortal),
    allTokens({
        cesiumCulledScreenSpaceError,
        cesiumEnableFogCulling,
        cesiumEnableFrustumCulling,
        cesiumEnforceCulledScreenSpaceError,
        cesiumForbidHoles,
        cesiumGeoreferenceOriginHeight,
        cesiumGeoreferenceOriginLatitude,
        cesiumGeoreferenceOriginLongitude,
        cesiumIonAccessToken,
        cesiumIonAssetId,
        cesiumLoadingDescendantLimit,
        cesiumMaximumCachedBytes,
        cesiumMaximumScreenSpaceError,
        cesiumMaximumSimultaneousTileLoads,
        cesiumPreloadAncestors,
        cesiumPreloadSiblings,
        cesiumProjectDefaultIonAccessToken,
        cesiumProjectDefaultIonAccessTokenId,
        cesiumSmoothNormals,
        cesiumSourceType,
        cesiumSuspendUpdate,
        cesiumUrl,
        ion,
        url
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
