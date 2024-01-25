#pragma once

#include "cesium/omniverse/AssetTroubleshootingDetails.h"
#include "cesium/omniverse/SetDefaultTokenResult.h"
#include "cesium/omniverse/TokenTroubleshootingDetails.h"

#include <pxr/usd/sdf/path.h>

#include <memory>

namespace CesiumIonClient {
struct Token;
}

namespace cesium::omniverse {

class CesiumIonSession;
class Context;
class OmniIonServer;

class CesiumIonServerManager {
  public:
    CesiumIonServerManager(Context* pContext);
    ~CesiumIonServerManager() = default;
    CesiumIonServerManager(const CesiumIonServerManager&) = delete;
    CesiumIonServerManager& operator=(const CesiumIonServerManager&) = delete;
    CesiumIonServerManager(CesiumIonServerManager&&) noexcept = delete;
    CesiumIonServerManager& operator=(CesiumIonServerManager&&) noexcept = delete;

    void onUpdateFrame();
    void setProjectDefaultToken(const CesiumIonClient::Token& token);
    void connectToIon();
    [[nodiscard]] OmniIonServer* getCurrentIonServer() const;
    [[nodiscard]] std::shared_ptr<CesiumIonSession> getCurrentIonSession() const;
    [[nodiscard]] std::optional<CesiumIonClient::Token> getDefaultToken() const;
    [[nodiscard]] SetDefaultTokenResult getSetDefaultTokenResult() const;
    [[nodiscard]] bool isDefaultTokenSet() const;
    void createToken(const std::string& name);
    void selectToken(const CesiumIonClient::Token& token);
    void specifyToken(const std::string& token);

    [[nodiscard]] std::optional<AssetTroubleshootingDetails> getAssetTroubleshootingDetails() const;
    [[nodiscard]] std::optional<TokenTroubleshootingDetails> getAssetTokenTroubleshootingDetails() const;
    [[nodiscard]] std::optional<TokenTroubleshootingDetails> getDefaultTokenTroubleshootingDetails() const;
    void updateTroubleshootingDetails(
        const pxr::SdfPath& tilesetPath,
        int64_t tilesetIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId);
    void updateTroubleshootingDetails(
        const pxr::SdfPath& tilesetPath,
        [[maybe_unused]] int64_t tilesetIonAssetId,
        int64_t imageryIonAssetId,
        uint64_t tokenEventId,
        uint64_t assetEventId);
    void updateTokenTroubleshootingDetails(
        int64_t assetId,
        const std::string& token,
        uint64_t eventId,
        TokenTroubleshootingDetails& details);
    void updateAssetTroubleshootingDetails(int64_t assetId, uint64_t eventId, AssetTroubleshootingDetails& details);

  private:
    Context* _pContext;
    SetDefaultTokenResult _lastSetTokenResult;

    std::optional<AssetTroubleshootingDetails> _assetTroubleshootingDetails;
    std::optional<TokenTroubleshootingDetails> _assetTokenTroubleshootingDetails;
    std::optional<TokenTroubleshootingDetails> _defaultTokenTroubleshootingDetails;
};

} // namespace cesium::omniverse
