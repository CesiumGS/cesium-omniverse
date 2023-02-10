#include "cesium/omniverse/TokenTroubleshooter.h"

#include "cesium/omniverse/OmniTileset.h"

namespace cesium::omniverse {
void TokenTroubleshooter::updateTokenTroubleshootingDetails(int tokenEventId) {
    auto session = OmniTileset::getSession();
    if (!session.has_value()) {
        // TODO: Signal an error.
        return;
    }

    auto connection = session.value()->getConnection();
    if (!connection.has_value()) {
        // TODO: Signal an error.
        return;
    }

    connection->me().thenInMainThread(
        [connection, tokenEventId](CesiumIonClient::Response<CesiumIonClient::Profile>&& profile) {
            
        });
}
void TokenTroubleshooter::updateAssetTroubleshootingDetails(int assetEventId) {
    auto session = OmniTileset::getSession();
    if (!session.has_value()) {
        return;
    }
}
} // namespace cesium::omniverse
