#pragma once

#include <omni/kit/IApp.h>
#include <carb/events/IEvents.h>

const char* ASSETS_UPDATED_EVENT_KEY = "cesium.omniverse.ASSETS_UPDATED";
const char* CONNECTION_UPDATED_EVENT_KEY = "cesium.omniverse.CONNECTION_UPDATED";
const char* PROFILE_UPDATED_EVENT_KEY = "cesium.omniverse.PROFILE_UPDATED";
const char* TOKENS_UPDATED_EVENT_KEY = "cesium.omniverse.TOKENS_UPDATED";

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
  private:
    static void sendMessageToBus(const char* eventKey) {
        auto eventType = carb::events::typeFromString(eventKey);
        auto app = carb::getCachedInterface<omni::kit::IApp>();
        auto bus = app->getMessageBusEventStream();
        bus->push(eventType);
    }
};


} // namespace cesium::omniverse
