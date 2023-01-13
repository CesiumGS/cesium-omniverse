#include "../include/cesium/omniverse/tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    height("height", TfToken::Immortal),
    latitude("latitude", TfToken::Immortal),
    longitude("longitude", TfToken::Immortal),
    allTokens({
        height,
        latitude,
        longitude
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
