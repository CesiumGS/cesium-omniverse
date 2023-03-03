#include "cesium/omniverse/UsdNotificationHandler.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

namespace cesium::omniverse {

namespace {
ChangedPrimType getType(const pxr::SdfPath& path) {
    if (UsdUtil::primExists(path)) {
        if (UsdUtil::isCesiumData(path)) {
            return ChangedPrimType::CESIUM_DATA;
        } else if (UsdUtil::isCesiumTileset(path)) {
            return ChangedPrimType::CESIUM_TILESET;
        } else if (UsdUtil::isCesiumRasterOverlay(path)) {
            return ChangedPrimType::CESIUM_RASTER_OVERLAY;
        }
    } else {
        auto item = AssetRegistry::getInstance().getItemByPath(path);

        if (item.has_value()) {
            switch (item.value().type) {
                case AssetType::TILESET:
                    return ChangedPrimType::CESIUM_TILESET;
                case AssetType::IMAGERY:
                    return ChangedPrimType::CESIUM_RASTER_OVERLAY;
                default:
                    break;
            }
        }
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

std::vector<ChangedPrim> UsdNotificationHandler::popChangedPrims() {
    static const std::vector<ChangedPrim> empty;

    if (_changedProperties.size() == 0) {
        return empty;
    }

    std::vector<ChangedPrim> changedProperties;

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
        auto type = getType(path);
        if (UsdUtil::isCesiumTileset(path)) {
            // TODO: implement prim add
        } else if (type != ChangedPrimType::OTHER) {
            // isCesiumTileset detects if the prim still exists. If it doesn't, we can use getType to determine the type
            //   that was deleted. If it's anything else other than ChangedPrimType::OTHER, we need to handle it.
            _changedProperties.emplace_back(ChangedPrim{path, pxr::TfToken(""), type, ChangeType::PRIM_REMOVED});
        }
    }

    const auto& changedProperties = objectsChanged.GetChangedInfoOnlyPaths();
    for (const auto& propertyPath : changedProperties) {
        const auto& name = propertyPath.GetNameToken();
        const auto& primPath = propertyPath.GetPrimPath();
        const auto& type = getType(primPath);
        if (type != ChangedPrimType::OTHER) {
            _changedProperties.emplace_back(ChangedPrim{primPath, name, type, ChangeType::PROPERTY_CHANGED});
            CESIUM_LOG_INFO("Changed property: {}", propertyPath.GetText());
        }
    }
}
} // namespace cesium::omniverse
