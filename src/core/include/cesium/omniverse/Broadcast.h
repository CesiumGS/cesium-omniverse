#pragma once

#include <carb/events/IEvents.h>
#include <omni/kit/IApp.h>

// TODO: We probably should put these constants somewhere else.
inline const char* ASSETS_UPDATED_EVENT_KEY = "cesium.omniverse.ASSETS_UPDATED";
inline const char* CONNECTION_UPDATED_EVENT_KEY = "cesium.omniverse.CONNECTION_UPDATED";
inline const char* PROFILE_UPDATED_EVENT_KEY = "cesium.omniverse.PROFILE_UPDATED";
inline const char* TOKENS_UPDATED_EVENT_KEY = "cesium.omniverse.TOKENS_UPDATED";
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
    static void setDefaultTokenComplete() {
        sendMessageToBus(SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY);
    }

  private:
    static void sendMessageToBus(const char* eventKey) {
        auto eventType = carb::events::typeFromString(eventKey);
        auto app = carb::getCachedInterface<omni::kit::IApp>();
        auto bus = app->getMessageBusEventStream();
        bus->push(eventType);
    }
};

} // namespace cesium::omniverse
