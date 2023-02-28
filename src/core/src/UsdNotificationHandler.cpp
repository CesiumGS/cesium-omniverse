#include "cesium/omniverse/UsdNotificationHandler.h"

#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {

namespace {
ChangedPrimType getType(const pxr::SdfPath& path) {
    if (UsdUtil::isCesiumData(path)) {
        return ChangedPrimType::CESIUM_DATA;
    } else if (UsdUtil::isCesiumTileset(path)) {
        return ChangedPrimType::CESIUM_TILESET;
    } else if (UsdUtil::isCesiumRasterOverlay(path)) {
        return ChangedPrimType::CESIUM_RASTER_OVERLAY;
    }

    return ChangedPrimType::OTHER;
}
} // namespace

UsdNotificationHandler::UsdNotificationHandler() {
    _noticeListenerKey = pxr::TfNotice::Register(pxr::TfCreateWeakPtr(this), &UsdNotificationHandler::onObjectsChanged);
}

UsdNotificationHandler::~UsdNotificationHandler() {
    pxr::TfNotice::Revoke(_noticeListenerKey);
}

std::vector<ChangedProperty> UsdNotificationHandler::popChangedProperties() {
    static const std::vector<ChangedProperty> empty;

    if (_changedProperties.size() == 0) {
        return empty;
    }

    std::vector<ChangedProperty> changedProperties;

    changedProperties.insert(
        changedProperties.end(),
        std::make_move_iterator(_changedProperties.begin()),
        std::make_move_iterator(_changedProperties.end()));

    _changedProperties.erase(_changedProperties.begin(), _changedProperties.end());

    return changedProperties;
}

void UsdNotificationHandler::onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged) {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto& resyncedPaths = objectsChanged.GetResyncedPaths();
    for (auto& path : resyncedPaths) {
        (void)path;
        // TODO: implement prim add, removal, deletion
    }

    const auto& changedProperties = objectsChanged.GetChangedInfoOnlyPaths();
    for (const auto& propertyPath : changedProperties) {
        const auto& name = propertyPath.GetNameToken();
        const auto& primPath = propertyPath.GetPrimPath();
        const auto& type = getType(primPath);
        if (type != ChangedPrimType::OTHER) {
            _changedProperties.emplace_back(ChangedProperty{primPath, name, type});
            CESIUM_LOG_INFO("Changed property: {}", propertyPath.GetText());
        }
    }
}
} // namespace cesium::omniverse
