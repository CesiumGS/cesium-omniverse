#include "cesium/omniverse/TokenTroubleshooter.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/OmniTileset.h"

namespace cesium::omniverse {
TokenTroubleshooter::TokenTroubleshooter(const std::shared_ptr<OmniTileset>& asset) {
    this->tileset = asset;
}

void TokenTroubleshooter::updateTokenTroubleshootingDetails(uint64_t eventId, TokenTroubleshootingDetails& details) {
    auto session = OmniTileset::getSession();
    if (!session.has_value()) {
        // TODO: Signal an error.
        return;
    }

    auto token = tileset->getTilesetToken();
    if (!token.has_value()) {
        // TODO: Figure out what we are supposed to do in this case. Error?
        return;
    }

    auto assetId = tileset->getIonAssetId();
    if (assetId < 1) {
        Broadcast::sendMessageToBus(eventId);
        return;
    }

    auto connection = std::make_shared<CesiumIonClient::Connection>(
        session.value()->getAsyncSystem(), session.value()->getAssetAccessor(), token->token);

    connection->me()
        .thenInMainThread(
            [assetId, connection, &details](CesiumIonClient::Response<CesiumIonClient::Profile>&& profile) {
                details.isValid = profile.value.has_value();
                return connection->asset(assetId);
            })
        .thenInMainThread([connection, &details](CesiumIonClient::Response<CesiumIonClient::Asset>&& asset) {
            details.allowsAccessToAsset = asset.value.has_value();

            auto ionSession = OmniTileset::getSession().value();
            ionSession->resume();
            const std::optional<CesiumIonClient::Connection>& userConnection = ionSession->getConnection();
            if (!userConnection) {
                CesiumIonClient::Response<CesiumIonClient::TokenList> result{};
                return ionSession->getAsyncSystem().createResolvedFuture(std::move(result));
            }

            return userConnection->tokens();
        })
        .thenInMainThread(
            [connection, &details, eventId](CesiumIonClient::Response<CesiumIonClient::TokenList>&& tokens) {
                if (tokens.value.has_value()) {
                    auto it = std::find_if(
                        tokens.value->items.begin(),
                        tokens.value->items.end(),
                        [&connection](const CesiumIonClient::Token& token) {
                            return token.token == connection->getAccessToken();
                        });
                    details.associatedWithUserAccount = it != tokens.value->items.end();
                }

                Broadcast::sendMessageToBus(eventId);
            });
}
void TokenTroubleshooter::updateAssetTroubleshootingDetails(uint64_t eventId, AssetTroubleshootingDetails& details) {
    auto session = OmniTileset::getSession();
    if (!session.has_value()) {
        return;
    }

    auto assetId = tileset->getIonAssetId();
    if (assetId < 1) {
        Broadcast::sendMessageToBus(eventId);
        return;
    }

    session.value()->getConnection()->asset(assetId).thenInMainThread(
        [eventId, &details](CesiumIonClient::Response<CesiumIonClient::Asset>&& asset) {
            details.assetExistsInUserAccount = asset.value.has_value();

            Broadcast::sendMessageToBus(eventId);
        });
}
} // namespace cesium::omniverse
