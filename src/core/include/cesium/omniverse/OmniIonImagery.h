#pragma once

#include "cesium/omniverse/OmniImagery.h"

#include <CesiumIonClient/Token.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniIonImagery : public OmniImagery {
  public:
    OmniIonImagery(const pxr::SdfPath& path);

    [[nodiscard]] std::string getIonApiUrl() const;
    [[nodiscard]] pxr::SdfPath getIonServerPath() const;
    [[nodiscard]] int64_t getIonAssetId() const;
    [[nodiscard]] std::optional<CesiumIonClient::Token> getIonAccessToken() const;
};
} // namespace cesium::omniverse
