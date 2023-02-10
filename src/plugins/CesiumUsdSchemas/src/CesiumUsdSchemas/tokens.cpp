#include ".//tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    cesiumAssetId("cesium:assetId", TfToken::Immortal),
    cesiumAssetUrl("cesium:assetUrl", TfToken::Immortal),
    cesiumDefaultProjectToken("cesium:defaultProjectToken", TfToken::Immortal),
    cesiumDefaultProjectTokenId("cesium:defaultProjectTokenId", TfToken::Immortal),
    cesiumGeoreferenceOrigin("cesium:georeferenceOrigin", TfToken::Immortal),
    allTokens({
        cesiumAssetId,
        cesiumAssetUrl,
        cesiumDefaultProjectToken,
        cesiumDefaultProjectTokenId,
        cesiumGeoreferenceOrigin
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
