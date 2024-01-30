#pragma once

#include <pxr/usd/usd/notice.h>

namespace cesium::omniverse {

class Context;

class UsdNotificationHandler final : public pxr::TfWeakBase {
  public:
    UsdNotificationHandler(Context* pContext);
    ~UsdNotificationHandler();
    UsdNotificationHandler(const UsdNotificationHandler&) = delete;
    UsdNotificationHandler& operator=(const UsdNotificationHandler&) = delete;
    UsdNotificationHandler(UsdNotificationHandler&&) noexcept = delete;
    UsdNotificationHandler& operator=(UsdNotificationHandler&&) noexcept = delete;

    void onStageLoaded();
    void onUpdateFrame();
    void clear();

  private:
    enum class ChangedPrimType {
        CESIUM_DATA,
        CESIUM_TILESET,
        CESIUM_ION_RASTER_OVERLAY,
        CESIUM_POLYGON_RASTER_OVERLAY,
        CESIUM_GEOREFERENCE,
        CESIUM_GLOBE_ANCHOR,
        CESIUM_ION_SERVER,
        CESIUM_CARTOGRAPHIC_POLYGON,
        USD_SHADER,
        OTHER,
    };

    enum class ChangedType {
        PROPERTY_CHANGED,
        PRIM_ADDED,
        PRIM_REMOVED,
    };

    struct ChangedPrim {
        pxr::SdfPath primPath;
        std::vector<pxr::TfToken> properties;
        ChangedPrimType primType;
        ChangedType changedType;
    };

    bool processChangedPrims();
    [[nodiscard]] bool processChangedPrim(const ChangedPrim& changedPrim) const;

    bool alreadyRegistered(const pxr::SdfPath& path);

    void onObjectsChanged(const pxr::UsdNotice::ObjectsChanged& objectsChanged);
    void onPrimAdded(const pxr::SdfPath& path);
    void onPrimRemoved(const pxr::SdfPath& path);
    void onPropertyChanged(const pxr::SdfPath& path);

    void insertAddedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType);
    void insertRemovedPrim(const pxr::SdfPath& primPath, ChangedPrimType primType);
    void
    insertPropertyChanged(const pxr::SdfPath& primPath, ChangedPrimType primType, const pxr::TfToken& propertyName);

    ChangedPrimType getTypeFromStage(const pxr::SdfPath& path) const;
    ChangedPrimType getTypeFromAssetRegistry(const pxr::SdfPath& path) const;

    Context* _pContext;
    pxr::TfNotice::Key _noticeListenerKey;
    std::vector<ChangedPrim> _changedPrims;
};

} // namespace cesium::omniverse
