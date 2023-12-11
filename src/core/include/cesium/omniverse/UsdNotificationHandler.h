#pragma once

#include <pxr/usd/usd/notice.h>

namespace cesium::omniverse {

class AssetRegistry;

enum class PrimType {
    CESIUM_DATA,
    CESIUM_TILESET,
    CESIUM_IMAGERY,
    CESIUM_GEOREFERENCE,
    CESIUM_GLOBE_ANCHOR,
    CESIUM_ION_SERVER,
    USD_SHADER,
    OTHER,
};

struct AddedPrim {
    pxr::SdfPath primPath;
    PrimType primType;
};

struct RemovedPrim {
    pxr::SdfPath primPath;
    PrimType primType;
};

struct PropertyChangedPrim {
    pxr::SdfPath primPath;
    PrimType primType;
    pxr::TfToken propertyName;
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

    void insertAddedPrim(const pxr::SdfPath& primPath, PrimType primType);
    void insertRemovedPrim(const pxr::SdfPath& primPath, PrimType primType);
    void insertPropertyChangedPrim(const pxr::SdfPath& primPath, PrimType primType, const pxr::TfToken& propertyName);

    pxr::TfNotice::Key _noticeListenerKey;

    std::vector<AddedPrim> _addedPrims;
    std::vector<RemovedPrim> _removedPrims;
    std::vector<PropertyChangedPrim> _propertyChangedPrims;
};

} // namespace cesium::omniverse
