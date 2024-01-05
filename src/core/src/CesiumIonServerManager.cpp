#include "cesium/omniverse/CesiumIonServerManager.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/OmniData.h"
#include "cesium/omniverse/OmniIonImagery.h"
#include "cesium/omniverse/OmniIonServer.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/TokenTroubleshootingDetails.h"

#include <CesiumIonClient/Connection.h>

namespace cesium::omniverse {

CesiumIonServerManager::CesiumIonServerManager(Context* pContext)
    : _pContext(pContext) {}

void CesiumIonServerManager::onUpdateFrame() {
    const auto& ionServers = _pContext->getAssetRegistry().getAllIonServers();

    for (const auto& pIonServer : ionServers) {
        pIonServer->getSession()->tick();
    }
}

void CesiumIonServerManager::setProjectDefaultToken(const CesiumIonClient::Token& token) {
    if (token.token.empty()) {
        return;
    }

    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return;
    }

    pCurrentIonServer->setToken(token);
}

void CesiumIonServerManager::updateTokenTroubleshootingDetails(
    int64_t ionAssetId,
    const std::string& token,
    uint64_t eventId,
    TokenTroubleshootingDetails& details) {
    const auto pSession = getCurrentIonSession();
    if (!pSession) {
        // TODO: Signal an error.
        return;
    }

    details.showDetails = true;

    const auto pConnection =
        std::make_shared<CesiumIonClient::Connection>(pSession->getAsyncSystem(), pSession->getAssetAccessor(), token);

    pConnection->me()
        .thenInMainThread(
            [ionAssetId, pConnection, &details](CesiumIonClient::Response<CesiumIonClient::Profile>&& profile) {
                details.isValid = profile.value.has_value();
                return pConnection->asset(ionAssetId);
            })
        .thenInMainThread([this, pConnection, &details](CesiumIonClient::Response<CesiumIonClient::Asset>&& asset) {
            details.allowsAccessToAsset = asset.value.has_value();

            const auto pIonSession = getCurrentIonSession();
            pIonSession->resume();
            const std::optional<CesiumIonClient::Connection>& userConnection = pIonSession->getConnection();
            if (!userConnection) {
                CesiumIonClient::Response<CesiumIonClient::TokenList> result{};
                return pIonSession->getAsyncSystem().createResolvedFuture(std::move(result));
            }

            return userConnection.value().tokens();
        })
        .thenInMainThread(
            [pConnection, &details, eventId](CesiumIonClient::Response<CesiumIonClient::TokenList>&& tokens) {
                if (tokens.value.has_value()) {
                    details.associatedWithUserAccount = CppUtil::containsByMember(
                        tokens.value.value().items, &CesiumIonClient::Token::token, pConnection->getAccessToken());
                }

                Broadcast::sendMessageToBus(eventId);
            });
}
void CesiumIonServerManager::updateAssetTroubleshootingDetails(
    int64_t ionAssetId,
    uint64_t eventId,
    AssetTroubleshootingDetails& details) {
    const auto pSession = getCurrentIonSession();
    if (!pSession) {
        return;
    }

    pSession->getConnection()
        ->asset(ionAssetId)
        .thenInMainThread([eventId, &details](CesiumIonClient::Response<CesiumIonClient::Asset>&& asset) {
            details.assetExistsInUserAccount = asset.value.has_value();

            Broadcast::sendMessageToBus(eventId);
        });
}

OmniIonServer* CesiumIonServerManager::getCurrentIonServer() const {
    const auto pData = _pContext->getAssetRegistry().getFirstData();

    if (!pData) {
        return _pContext->getAssetRegistry().getFirstIonServer();
    }

    const auto selectedIonServerPath = pData->getSelectedIonServerPath();

    if (selectedIonServerPath.IsEmpty()) {
        return _pContext->getAssetRegistry().getFirstIonServer();
    }

    const auto pIonServer = _pContext->getAssetRegistry().getIonServer(selectedIonServerPath);

    if (!pIonServer) {
        return _pContext->getAssetRegistry().getFirstIonServer();
    }

    return pIonServer;
}

void CesiumIonServerManager::connectToIon() {
    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return;
    }

    pCurrentIonServer->getSession()->connect();
}

std::shared_ptr<CesiumIonSession> CesiumIonServerManager::getCurrentIonSession() const {
    // A lot of UI code will end up calling the session prior to us actually having a stage. The user won't see this
    // but some major segfaults will occur without this check.
    if (!_pContext->hasUsdStage()) {
        return nullptr;
    }

    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return nullptr;
    }

    return pCurrentIonServer->getSession();
}

std::optional<CesiumIonClient::Token> CesiumIonServerManager::getDefaultToken() const {
    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return std::nullopt;
    }

    const auto token = pCurrentIonServer->getToken();

    if (token.token.empty()) {
        return std::nullopt;
    }

    return token;
}

SetDefaultTokenResult CesiumIonServerManager::getSetDefaultTokenResult() const {
    return _lastSetTokenResult;
}

bool CesiumIonServerManager::isDefaultTokenSet() const {
    return getDefaultToken().has_value();
}

void CesiumIonServerManager::createToken(const std::string& name) {
    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return;
    }

    const auto pConnection = pCurrentIonServer->getSession()->getConnection();

    if (!pConnection.has_value()) {
        _lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::NOT_CONNECTED_TO_ION,
            std::string(SetDefaultTokenResultMessages::NOT_CONNECTED_TO_ION_MESSAGE),
        };
        return;
    }

    pConnection->createToken(name, {"assets:read"}, std::vector<int64_t>{1}, std::nullopt)
        .thenInMainThread([this](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
            if (response.value) {
                setProjectDefaultToken(response.value.value());

                _lastSetTokenResult = SetDefaultTokenResult{
                    SetDefaultTokenResultCode::OK,
                    std::string(SetDefaultTokenResultMessages::OK_MESSAGE),
                };
            } else {
                _lastSetTokenResult = SetDefaultTokenResult{
                    SetDefaultTokenResultCode::CREATE_FAILED,
                    fmt::format(
                        SetDefaultTokenResultMessages::CREATE_FAILED_MESSAGE_BASE,
                        response.errorMessage,
                        response.errorCode),
                };
            }

            Broadcast::setDefaultTokenComplete();
        });
}
void CesiumIonServerManager::selectToken(const CesiumIonClient::Token& token) {
    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return;
    }

    const auto& connection = pCurrentIonServer->getSession()->getConnection();

    if (!connection.has_value()) {
        _lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::NOT_CONNECTED_TO_ION,
            std::string(SetDefaultTokenResultMessages::NOT_CONNECTED_TO_ION_MESSAGE),
        };
    } else {
        setProjectDefaultToken(token);

        _lastSetTokenResult = SetDefaultTokenResult{
            SetDefaultTokenResultCode::OK,
            std::string(SetDefaultTokenResultMessages::OK_MESSAGE),
        };
    }

    Broadcast::setDefaultTokenComplete();
}
void CesiumIonServerManager::specifyToken(const std::string& token) {
    const auto pCurrentIonServer = getCurrentIonServer();

    if (!pCurrentIonServer) {
        return;
    }

    const auto pSession = pCurrentIonServer->getSession();

    pSession->findToken(token).thenInMainThread(
        [this, token](CesiumIonClient::Response<CesiumIonClient::Token>&& response) {
            if (response.value) {
                setProjectDefaultToken(response.value.value());
            } else {
                CesiumIonClient::Token t;
                t.token = token;
                setProjectDefaultToken(t);
            }
            // We assume the user knows what they're doing if they specify a token not on their account.
            _lastSetTokenResult = SetDefaultTokenResult{
                SetDefaultTokenResultCode::OK,
                std::string(SetDefaultTokenResultMessages::OK_MESSAGE),
            };

            Broadcast::setDefaultTokenComplete();
        });
}

std::optional<AssetTroubleshootingDetails> CesiumIonServerManager::getAssetTroubleshootingDetails() const {
    return _assetTroubleshootingDetails;
}
std::optional<TokenTroubleshootingDetails> CesiumIonServerManager::getAssetTokenTroubleshootingDetails() const {
    return _assetTokenTroubleshootingDetails;
}
std::optional<TokenTroubleshootingDetails> CesiumIonServerManager::getDefaultTokenTroubleshootingDetails() const {
    return _defaultTokenTroubleshootingDetails;
}
void CesiumIonServerManager::updateTroubleshootingDetails(
    const PXR_NS::SdfPath& tilesetPath,
    int64_t tilesetIonAssetId,
    uint64_t tokenEventId,
    uint64_t assetEventId) {
    const auto pTileset = _pContext->getAssetRegistry().getTileset(tilesetPath);

    if (!pTileset) {
        return;
    }

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    updateAssetTroubleshootingDetails(tilesetIonAssetId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    const auto& defaultToken = getDefaultToken();
    if (defaultToken.has_value()) {
        const auto& token = defaultToken.value().token;
        updateTokenTroubleshootingDetails(
            tilesetIonAssetId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    _assetTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    auto tilesetIonAccessToken = pTileset->getIonAccessToken();
    if (!tilesetIonAccessToken.token.empty()) {
        updateTokenTroubleshootingDetails(
            tilesetIonAssetId, tilesetIonAccessToken.token, tokenEventId, _assetTokenTroubleshootingDetails.value());
    }
}
void CesiumIonServerManager::updateTroubleshootingDetails(
    const PXR_NS::SdfPath& tilesetPath,
    [[maybe_unused]] int64_t tilesetIonAssetId,
    int64_t imageryIonAssetId,
    uint64_t tokenEventId,
    uint64_t assetEventId) {
    const auto pTileset = _pContext->getAssetRegistry().getTileset(tilesetPath);
    if (!pTileset) {
        return;
    }

    const auto pIonImagery = _pContext->getAssetRegistry().getIonImageryByIonAssetId(imageryIonAssetId);
    if (!pIonImagery) {
        return;
    }

    _assetTroubleshootingDetails = AssetTroubleshootingDetails();
    updateAssetTroubleshootingDetails(imageryIonAssetId, assetEventId, _assetTroubleshootingDetails.value());

    _defaultTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    const auto& defaultToken = getDefaultToken();
    if (defaultToken.has_value()) {
        const auto& token = defaultToken.value().token;
        updateTokenTroubleshootingDetails(
            imageryIonAssetId, token, tokenEventId, _defaultTokenTroubleshootingDetails.value());
    }

    _assetTokenTroubleshootingDetails = TokenTroubleshootingDetails();

    auto imageryIonAccessToken = pIonImagery->getIonAccessToken();
    if (!imageryIonAccessToken.token.empty()) {
        updateTokenTroubleshootingDetails(
            imageryIonAssetId, imageryIonAccessToken.token, tokenEventId, _assetTokenTroubleshootingDetails.value());
    }
}

} // namespace cesium::omniverse
