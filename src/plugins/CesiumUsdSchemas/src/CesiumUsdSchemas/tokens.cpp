#include ".//tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    assetId("assetId", TfToken::Immortal),
    assetUrl("assetUrl", TfToken::Immortal),
    ionToken("ionToken", TfToken::Immortal),
    allTokens({
        assetId,
        assetUrl,
        ionToken
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
