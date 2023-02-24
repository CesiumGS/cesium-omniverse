#pragma once

#include <CesiumIonClient/Token.h>

namespace cesium::omniverse {
struct TokenTroubleshootingDetails {
    CesiumIonClient::Token token;
    bool isValid{false};
    bool allowsAccessToAsset{false};
    bool associatedWithUserAccount{false};
};

struct AssetTroubleshootingDetails {
    int64_t assetId;
    bool assetExistsInUserAccount{false};
};

class TokenTroubleshooter {
  public:
    void updateTokenTroubleshootingDetails(
        int64_t assetId,
        const std::string& token,
        uint64_t eventId,
        TokenTroubleshootingDetails& details);
    void updateAssetTroubleshootingDetails(int64_t assetId, uint64_t eventId, AssetTroubleshootingDetails& details);
};
} // namespace cesium::omniverse
