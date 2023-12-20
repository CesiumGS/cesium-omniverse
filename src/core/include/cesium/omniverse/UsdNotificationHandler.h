#pragma once

#include <pxr/usd/usd/notice.h>

namespace cesium::omniverse {

class AssetRegistry;

enum class ChangedPrimType {
    CESIUM_DATA,
    CESIUM_TILESET,
    CESIUM_IMAGERY,
    CESIUM_GEOREFERENCE,
    CESIUM_GLOBE_ANCHOR,
    CESIUM_ION_SERVER,
    USD_SHADER,
    OTHER,
};

enum class ChangeType {
    PROPERTY_CHANGED,
    PRIM_ADDED,
    PRIM_REMOVED,
};

struct ChangedPrim {
    pxr::SdfPath primPath;
    std::vector<pxr::TfToken> properties;
    ChangedPrimType primType;
    ChangeType changeType;
};

class UsdNotificationHandler final : public pxr::TfWeakBase {
  public:
    UsdNotificationHandler();
    ~UsdNotificationHandler();

    void onStageLoaded();
    void onUpdateFrame();

  private:
    void onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged);
    void onPrimAdded(const pxr::SdfPath& path);
    void onPrimRemoved(const pxr::SdfPath& path);
    void onPropertyChanged(const pxr::SdfPath& path);

    void insertAddedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType);
    void insertRemovedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType);
    void
    insertPropertyChanged(const pxr::SdfPath& primPath, ChangedPrimType primType, const pxr::TfToken& propertyName);

    pxr::TfNotice::Key _noticeListenerKey;
    std::vector<ChangedPrim> _changedPrims;
};

} // namespace cesium::omniverse
