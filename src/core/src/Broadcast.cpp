#include "cesium/omniverse/Broadcast.h"
namespace cesium::omniverse::Broadcast {
namespace {
const char* ASSETS_UPDATED_EVENT_KEY = "cesium.omniverse.ASSETS_UPDATED";
const char* CONNECTION_UPDATED_EVENT_KEY = "cesium.omniverse.CONNECTION_UPDATED";
const char* PROFILE_UPDATED_EVENT_KEY = "cesium.omniverse.PROFILE_UPDATED";
const char* TOKENS_UPDATED_EVENT_KEY = "cesium.omniverse.TOKENS_UPDATED";
const char* SHOW_TROUBLESHOOTER_EVENT_KEY = "cesium.omniverse.SHOW_TROUBLESHOOTER";
const char* SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY = "cesium.omniverse.SET_DEFAULT_PROJECT_TOKEN_COMPLETE";

template <typename... ValuesT>
void sendMessageToBusWithPayload(carb::events::EventType eventType, ValuesT&&... payload) {
    auto app = carb::getCachedInterface<omni::kit::IApp>();
    auto bus = app->getMessageBusEventStream();
    bus->push(eventType, std::forward<ValuesT>(payload)...);
}

template <typename... ValuesT> void sendMessageToBusWithPayload(const char* eventKey, ValuesT&&... payload) {
    auto eventType = carb::events::typeFromString(eventKey);
    sendMessageToBusWithPayload(eventType, std::forward<ValuesT>(payload)...);
}

} // namespace

void assetsUpdated() {
    sendMessageToBus(ASSETS_UPDATED_EVENT_KEY);
}

void connectionUpdated() {
    sendMessageToBus(CONNECTION_UPDATED_EVENT_KEY);
}

void profileUpdated() {
    sendMessageToBus(PROFILE_UPDATED_EVENT_KEY);
}

void tokensUpdated() {
    sendMessageToBus(TOKENS_UPDATED_EVENT_KEY);
}

void showTroubleshooter(
    int64_t tilesetAssetId,
    int64_t tilesetIonId,
    const std::string& tilesetName,
    int64_t rasterOverlayId,
    const std::string& rasterOverlayName,
    const std::string& message) {
    sendMessageToBusWithPayload(
        SHOW_TROUBLESHOOTER_EVENT_KEY,
        std::make_pair("tilesetAssetId", tilesetAssetId),
        std::make_pair("tilesetIonId", tilesetIonId),
        std::make_pair("tilesetName", tilesetName.c_str()),
        std::make_pair("rasterOverlayId", rasterOverlayId),
        std::make_pair("rasterOverlayName", rasterOverlayName.c_str()),
        std::make_pair("message", message.c_str()));
}

void setDefaultTokenComplete() {
    sendMessageToBus(SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY);
}

void sendMessageToBus(carb::events::EventType eventType) {
    auto app = carb::getCachedInterface<omni::kit::IApp>();
    auto bus = app->getMessageBusEventStream();
    bus->push(eventType);
}

void sendMessageToBus(const char* eventKey) {
    auto eventType = carb::events::typeFromString(eventKey);
    sendMessageToBus(eventType);
}

} // namespace cesium::omniverse::Broadcast
