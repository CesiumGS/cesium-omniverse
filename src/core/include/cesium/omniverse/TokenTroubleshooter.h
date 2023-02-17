#pragma once

#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/OmniTileset.h"

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
    TokenTroubleshooter(const std::shared_ptr<OmniTileset>& asset);
    void updateTokenTroubleshootingDetails(
        int64_t assetId,
        std::string& token,
        uint64_t eventId,
        TokenTroubleshootingDetails& details);
    void updateAssetTroubleshootingDetails(int64_t assetId, uint64_t eventId, AssetTroubleshootingDetails& details);

  private:
    std::shared_ptr<OmniTileset> tileset;
};
} // namespace cesium::omniverse