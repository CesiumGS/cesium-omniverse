#pragma once

// clang-format off
// carb/events/IObject.h should include this
#include <utility>
// clang-format on

#include <carb/events/IEvents.h>
#include <pxr/usd/usd/common.h>

namespace cesium::omniverse::Broadcast {

void assetsUpdated();
void connectionUpdated();
void profileUpdated();
void tokensUpdated();
void showTroubleshooter(
    const pxr::SdfPath& tilesetPath,
    int64_t tilesetIonAssetId,
    const std::string& tilesetName,
    int64_t rasterOverlayIonAssetId,
    const std::string& rasterOverlayName,
    const std::string& message);
void setDefaultTokenComplete();
void tilesetLoaded(const pxr::SdfPath& tilesetPath);
void sendMessageToBus(carb::events::EventType eventType);
void sendMessageToBus(const std::string_view& eventKey);

} // namespace cesium::omniverse::Broadcast
