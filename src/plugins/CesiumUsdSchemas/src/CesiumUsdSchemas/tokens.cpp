#include ".//tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

CesiumTokensType::CesiumTokensType() :
    assetId("assetId", TfToken::Immortal),
    assetUrl("assetUrl", TfToken::Immortal),
    defaultProjectToken("defaultProjectToken", TfToken::Immortal),
    defaultProjectTokenId("defaultProjectTokenId", TfToken::Immortal),
    allTokens({
        assetId,
        assetUrl,
        defaultProjectToken,
        defaultProjectTokenId
    })
{
}

TfStaticData<CesiumTokensType> CesiumTokens;

PXR_NAMESPACE_CLOSE_SCOPE
