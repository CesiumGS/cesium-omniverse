#pragma once

#include <carb/events/IEvents.h>
#include <omni/kit/IApp.h>

// TODO: We probably should put these constants somewhere else.
inline const char* ASSETS_UPDATED_EVENT_KEY = "cesium.omniverse.ASSETS_UPDATED";
inline const char* CONNECTION_UPDATED_EVENT_KEY = "cesium.omniverse.CONNECTION_UPDATED";
inline const char* PROFILE_UPDATED_EVENT_KEY = "cesium.omniverse.PROFILE_UPDATED";
inline const char* TOKENS_UPDATED_EVENT_KEY = "cesium.omniverse.TOKENS_UPDATED";
inline const char* SHOW_TROUBLESHOOTER_EVENT_KEY = "cesium.omniverse.SHOW_TROUBLESHOOTER";
inline const char* SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY = "cesium.omniverse.SET_DEFAULT_PROJECT_TOKEN_COMPLETE";

namespace cesium::omniverse {

class Broadcast {
  public:
    static void assetsUpdated() {
        sendMessageToBus(ASSETS_UPDATED_EVENT_KEY);
    }
    static void connectionUpdated() {
        sendMessageToBus(CONNECTION_UPDATED_EVENT_KEY);
    }
    static void profileUpdated() {
        sendMessageToBus(PROFILE_UPDATED_EVENT_KEY);
    }
    static void tokensUpdated() {
        sendMessageToBus(TOKENS_UPDATED_EVENT_KEY);
    }
    static void showTroubleshooter(
        int64_t tilesetId,
        const std::string& tilesetName,
        int64_t rasterOverlayId,
        const std::string& rasterOverlayName,
        const std::string& message) {
        sendMessageToBusWithPayload(
            SHOW_TROUBLESHOOTER_EVENT_KEY,
            std::make_pair("tilesetId", tilesetId),
            std::make_pair("tilesetName", tilesetName.c_str()),
            std::make_pair("rasterOverlayId", rasterOverlayId),
            std::make_pair("rasterOverlayName", rasterOverlayName.c_str()),
            std::make_pair("message", message.c_str()));
    }
    static void setDefaultTokenComplete() {
        sendMessageToBus(SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY);
    }
    static void sendMessageToBus(const char* eventKey) {
        auto eventType = carb::events::typeFromString(eventKey);
        sendMessageToBus(eventType);
    }
    static void sendMessageToBus(carb::events::EventType eventType) {
        auto app = carb::getCachedInterface<omni::kit::IApp>();
        auto bus = app->getMessageBusEventStream();
        bus->push(eventType);
    }
    template <typename... ValuesT> static void sendMessageToBusWithPayload(const char* eventKey, ValuesT&&... payload) {
        auto eventType = carb::events::typeFromString(eventKey);
        sendMessageToBusWithPayload(eventType, std::forward<ValuesT>(payload)...);
    }
    template <typename... ValuesT>
    static void sendMessageToBusWithPayload(carb::events::EventType eventType, ValuesT&&... payload) {
        auto app = carb::getCachedInterface<omni::kit::IApp>();
        auto bus = app->getMessageBusEventStream();
        bus->push(eventType, std::forward<ValuesT>(payload)...);
    }
};

} // namespace cesium::omniverse
