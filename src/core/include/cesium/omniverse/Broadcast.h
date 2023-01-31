#pragma once

#include <omni/kit/IApp.h>
#include <carb/events/IEvents.h>

const char* CONNECTION_UPDATED_EVENT_KEY = "cesium.omniverse.CONNECTION_UPDATED";

namespace cesium::omniverse {

class Broadcast {
  public:
    static void connectionUpdated() {
        auto connectionUpdatedEvent = carb::events::typeFromString(CONNECTION_UPDATED_EVENT_KEY);
        sendMessageToBus(connectionUpdatedEvent);
    }
  private:
    static void sendMessageToBus(carb::events::EventType eventType) {
        auto app = carb::getCachedInterface<omni::kit::IApp>();
        auto bus = app->getMessageBusEventStream();
        bus->push(eventType);
    }
};


} // namespace cesium::omniverse
