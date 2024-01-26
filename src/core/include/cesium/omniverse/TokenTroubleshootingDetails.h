#pragma once

#include <CesiumIonClient/Token.h>

namespace cesium::omniverse {

struct TokenTroubleshootingDetails {
    CesiumIonClient::Token token;
    bool isValid{false};
    bool allowsAccessToAsset{false};
    bool associatedWithUserAccount{false};
    bool showDetails{false};
};

} // namespace cesium::omniverse
