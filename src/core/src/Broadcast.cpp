#include "cesium/omniverse/Broadcast.h"

#include <omni/kit/IApp.h>
#include <pxr/usd/sdf/path.h>
namespace cesium::omniverse::Broadcast {

namespace {

const std::string_view ASSETS_UPDATED_EVENT_KEY = "cesium.omniverse.ASSETS_UPDATED";
const std::string_view CONNECTION_UPDATED_EVENT_KEY = "cesium.omniverse.CONNECTION_UPDATED";
const std::string_view PROFILE_UPDATED_EVENT_KEY = "cesium.omniverse.PROFILE_UPDATED";
const std::string_view TOKENS_UPDATED_EVENT_KEY = "cesium.omniverse.TOKENS_UPDATED";
const std::string_view SHOW_TROUBLESHOOTER_EVENT_KEY = "cesium.omniverse.SHOW_TROUBLESHOOTER";
const std::string_view SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY = "cesium.omniverse.SET_DEFAULT_PROJECT_TOKEN_COMPLETE";
const std::string_view TILESET_LOADED_KEY = "cesium.omniverse.TILESET_LOADED";

template <typename... ValuesT>
void sendMessageToBusWithPayload(carb::events::EventType eventType, ValuesT&&... payload) {
    const auto iApp = carb::getCachedInterface<omni::kit::IApp>();
    const auto bus = iApp->getMessageBusEventStream();
    bus->push(eventType, std::forward<ValuesT>(payload)...);
}

template <typename... ValuesT>
void sendMessageToBusWithPayload(const std::string_view& eventKey, ValuesT&&... payload) {
    const auto eventType = carb::events::typeFromString(eventKey.data());
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
    const pxr::SdfPath& tilesetPath,
    int64_t tilesetIonAssetId,
    const std::string& tilesetName,
    int64_t rasterOverlayIonAssetId,
    const std::string& rasterOverlayName,
    const std::string& message) {
    sendMessageToBusWithPayload(
        SHOW_TROUBLESHOOTER_EVENT_KEY,
        std::make_pair("tilesetPath", tilesetPath.GetText()),
        std::make_pair("tilesetIonAssetId", tilesetIonAssetId),
        std::make_pair("tilesetName", tilesetName.c_str()),
        std::make_pair("rasterOverlayIonAssetId", rasterOverlayIonAssetId),
        std::make_pair("rasterOverlayName", rasterOverlayName.c_str()),
        std::make_pair("message", message.c_str()));
}

void setDefaultTokenComplete() {
    sendMessageToBus(SET_DEFAULT_PROJECT_TOKEN_COMPLETE_KEY);
}

void tilesetLoaded(const pxr::SdfPath& tilesetPath) {
    sendMessageToBusWithPayload(TILESET_LOADED_KEY, std::make_pair("tilesetPath", tilesetPath.GetText()));
}

void sendMessageToBus(carb::events::EventType eventType) {
    const auto iApp = carb::getCachedInterface<omni::kit::IApp>();
    const auto bus = iApp->getMessageBusEventStream();
    bus->push(eventType);
}

void sendMessageToBus(const std::string_view& eventKey) {
    const auto eventType = carb::events::typeFromString(eventKey.data());
    sendMessageToBus(eventType);
}

} // namespace cesium::omniverse::Broadcast
