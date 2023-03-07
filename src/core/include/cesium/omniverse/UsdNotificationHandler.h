#pragma once

#include <pxr/usd/usd/notice.h>

namespace cesium::omniverse {

class AssetRegistry;

enum class ChangedPrimType {
    CESIUM_TILESET,
    CESIUM_IMAGERY,
    CESIUM_DATA,
    OTHER,
};

enum class ChangeType { PROPERTY_CHANGED, PRIM_ADDED [[maybe_unused]], PRIM_MOVED [[maybe_unused]], PRIM_REMOVED };

struct ChangedPrim {
    pxr::SdfPath path;
    pxr::TfToken name;
    ChangedPrimType primType;
    ChangeType changeType;
};

class UsdNotificationHandler : public pxr::TfWeakBase {
  public:
    UsdNotificationHandler();
    ~UsdNotificationHandler();

    std::vector<ChangedPrim> popChangedPrims();

  private:
    void onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged);

    pxr::TfNotice::Key _noticeListenerKey;
    std::vector<ChangedPrim> _changedProperties;
};

} // namespace cesium::omniverse
