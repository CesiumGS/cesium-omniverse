#include ".//tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    cesiumCulledScreenSpaceError("cesium:culledScreenSpaceError", TfToken::Immortal),
    cesiumDefaultProjectIonAccessToken("cesium:defaultProjectIonAccessToken", TfToken::Immortal),
    cesiumDefaultProjectIonAccessTokenId("cesium:defaultProjectIonAccessTokenId", TfToken::Immortal),
    cesiumEnableFogCulling("cesium:enableFogCulling", TfToken::Immortal),
    cesiumEnableFrustumCulling("cesium:enableFrustumCulling", TfToken::Immortal),
    cesiumEnforceCulledScreenSpaceError("cesium:enforceCulledScreenSpaceError", TfToken::Immortal),
    cesiumForbidHoles("cesium:forbidHoles", TfToken::Immortal),
    cesiumGeoreferenceOrigin("cesium:georeferenceOrigin", TfToken::Immortal),
    cesiumIonAccessToken("cesium:ionAccessToken", TfToken::Immortal),
    cesiumIonAssetId("cesium:ionAssetId", TfToken::Immortal),
    cesiumLoadingDescendantLimit("cesium:loadingDescendantLimit", TfToken::Immortal),
    cesiumMaximumCachedBytes("cesium:maximumCachedBytes", TfToken::Immortal),
    cesiumMaximumScreenSpaceError("cesium:maximumScreenSpaceError", TfToken::Immortal),
    cesiumMaximumSimultaneousTileLoads("cesium:maximumSimultaneousTileLoads", TfToken::Immortal),
    cesiumPreloadAncestors("cesium:preloadAncestors", TfToken::Immortal),
    cesiumPreloadSiblings("cesium:preloadSiblings", TfToken::Immortal),
    cesiumSuspendUpdate("cesium:suspendUpdate", TfToken::Immortal),
    cesiumUrl("cesium:url", TfToken::Immortal),
    allTokens({
        cesiumCulledScreenSpaceError,
        cesiumDefaultProjectIonAccessToken,
        cesiumDefaultProjectIonAccessTokenId,
        cesiumEnableFogCulling,
        cesiumEnableFrustumCulling,
        cesiumEnforceCulledScreenSpaceError,
        cesiumForbidHoles,
        cesiumGeoreferenceOrigin,
        cesiumIonAccessToken,
        cesiumIonAssetId,
        cesiumLoadingDescendantLimit,
        cesiumMaximumCachedBytes,
        cesiumMaximumScreenSpaceError,
        cesiumMaximumSimultaneousTileLoads,
        cesiumPreloadAncestors,
        cesiumPreloadSiblings,
        cesiumSuspendUpdate,
        cesiumUrl
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
