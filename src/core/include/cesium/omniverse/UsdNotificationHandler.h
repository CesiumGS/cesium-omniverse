#pragma once

#include <pxr/usd/usd/notice.h>

namespace cesium::omniverse {

enum class ChangedPrimType {
    CESIUM_TILESET,
    CESIUM_RASTER_OVERLAY,
    CESIUM_DATA,
    OTHER,
};

struct ChangedProperty {
    pxr::SdfPath path;
    pxr::TfToken token;
    ChangedPrimType type;
};

class UsdNotificationHandler : public pxr::TfWeakBase {
  public:
    UsdNotificationHandler();
    ~UsdNotificationHandler();

    std::vector<ChangedProperty> popChangedProperties();

  private:
    void onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged);

    pxr::TfNotice::Key _noticeListenerKey;
    std::vector<ChangedProperty> _changedProperties;
};

} // namespace cesium::omniverse
