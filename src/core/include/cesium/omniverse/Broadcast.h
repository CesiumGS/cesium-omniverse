#pragma once

#include <carb/events/IEvents.h>
#include <omni/kit/IApp.h>

namespace cesium::omniverse::Broadcast {

void assetsUpdated();
void connectionUpdated();
void profileUpdated();
void tokensUpdated();
void showTroubleshooter(
    int64_t tilesetAssetId,
    int64_t tilesetIonId,
    const std::string& tilesetName,
    int64_t rasterOverlayId,
    const std::string& rasterOverlayName,
    const std::string& message);
void setDefaultTokenComplete();
void sendMessageToBus(carb::events::EventType eventType);
void sendMessageToBus(const char* eventKey);
} // namespace cesium::omniverse::Broadcast
