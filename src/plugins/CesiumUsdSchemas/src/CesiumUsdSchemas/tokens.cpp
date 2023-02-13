#include ".//tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    cesiumDefaultProjectToken("cesium:defaultProjectToken", TfToken::Immortal),
    cesiumDefaultProjectTokenId("cesium:defaultProjectTokenId", TfToken::Immortal),
    cesiumGeoreferenceOrigin("cesium:georeferenceOrigin", TfToken::Immortal),
    cesiumIonToken("cesium:ionToken", TfToken::Immortal),
    cesiumName("cesium:name", TfToken::Immortal),
    cesiumRasterOverlayId("cesium:rasterOverlayId", TfToken::Immortal),
    cesiumTilesetId("cesium:tilesetId", TfToken::Immortal),
    cesiumTilesetUrl("cesium:tilesetUrl", TfToken::Immortal),
    allTokens({
        cesiumDefaultProjectToken,
        cesiumDefaultProjectTokenId,
        cesiumGeoreferenceOrigin,
        cesiumIonToken,
        cesiumName,
        cesiumRasterOverlayId,
        cesiumTilesetId,
        cesiumTilesetUrl
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
