#pragma once

// carb/events/IObject.h should include this
#include <utility>

#include <carb/events/IEvents.h>
#include <omni/kit/IApp.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse::Broadcast {

void assetsUpdated();
void connectionUpdated();
void profileUpdated();
void tokensUpdated();
void showTroubleshooter(
    const pxr::SdfPath& tilesetPath,
    int64_t tilesetIonAssetId,
    const std::string& tilesetName,
    int64_t imageryIonAssetId,
    const std::string& imageryName,
    const std::string& message);
void setDefaultTokenComplete();
void tilesetLoaded(const pxr::SdfPath& tilesetPath);
void sendMessageToBus(carb::events::EventType eventType);
void sendMessageToBus(const char* eventKey);
} // namespace cesium::omniverse::Broadcast
