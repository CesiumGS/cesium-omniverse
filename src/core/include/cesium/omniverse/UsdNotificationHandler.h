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

enum class ChangeType {
    PROPERTY_CHANGED,
    PRIM_ADDED,
    PRIM_REMOVED,
};

struct ChangedPrim {
    pxr::SdfPath path;
    pxr::TfToken propertyName;
    ChangedPrimType primType;
    ChangeType changeType;
};

class UsdNotificationHandler final : public pxr::TfWeakBase {
  public:
    UsdNotificationHandler();
    ~UsdNotificationHandler();

    std::vector<ChangedPrim> popChangedPrims();

  private:
    void onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged);
    void onPrimAdded(const pxr::SdfPath& path);
    void onPrimRemoved(const pxr::SdfPath& path);
    void onPropertyChanged(const pxr::SdfPath& path);

    pxr::TfNotice::Key _noticeListenerKey;
    std::vector<ChangedPrim> _changedPrims;
};

} // namespace cesium::omniverse
