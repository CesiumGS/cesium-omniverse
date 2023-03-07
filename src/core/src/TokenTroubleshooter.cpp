#include "cesium/omniverse/TokenTroubleshooter.h"

#include "cesium/omniverse/Broadcast.h"
#include "cesium/omniverse/CesiumIonSession.h"
#include "cesium/omniverse/Context.h"

#include <CesiumIonClient/Connection.h>

namespace cesium::omniverse {
void TokenTroubleshooter::updateTokenTroubleshootingDetails(
    int64_t ionAssetId,
    const std::string& token,
    uint64_t eventId,
    TokenTroubleshootingDetails& details) {
    auto session = Context::instance().getSession();
    if (!session.has_value()) {
        // TODO: Signal an error.
        return;
    }

    details.showDetails = true;

    auto connection = std::make_shared<CesiumIonClient::Connection>(
        session.value()->getAsyncSystem(), session.value()->getAssetAccessor(), token);

    connection->me()
        .thenInMainThread(
            [ionAssetId, connection, &details](CesiumIonClient::Response<CesiumIonClient::Profile>&& profile) {
                details.isValid = profile.value.has_value();
                return connection->asset(ionAssetId);
            })
        .thenInMainThread([connection, &details](CesiumIonClient::Response<CesiumIonClient::Asset>&& asset) {
            details.allowsAccessToAsset = asset.value.has_value();

            auto ionSession = Context::instance().getSession().value();
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
void TokenTroubleshooter::updateAssetTroubleshootingDetails(
    int64_t ionAssetId,
    uint64_t eventId,
    AssetTroubleshootingDetails& details) {
    auto session = Context::instance().getSession();
    if (!session.has_value()) {
        return;
    }

    session.value()->getConnection()->asset(ionAssetId).thenInMainThread(
        [eventId, &details](CesiumIonClient::Response<CesiumIonClient::Asset>&& asset) {
            details.assetExistsInUserAccount = asset.value.has_value();

            Broadcast::sendMessageToBus(eventId);
        });
}
} // namespace cesium::omniverse
