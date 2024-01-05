#pragma once

#include <pxr/usd/usd/notice.h>

namespace cesium::omniverse {

class Context;

class UsdNotificationHandler final : public PXR_NS::TfWeakBase {
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
        CESIUM_ION_IMAGERY,
        CESIUM_POLYGON_IMAGERY,
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
        PXR_NS::SdfPath primPath;
        std::vector<PXR_NS::TfToken> properties;
        ChangedPrimType primType;
        ChangedType changedType;
    };

    bool processChangedPrims();
    [[nodiscard]] bool processChangedPrim(const ChangedPrim& changedPrim) const;

    bool alreadyRegistered(const PXR_NS::SdfPath& path);

    void onObjectsChanged(const PXR_NS::UsdNotice::ObjectsChanged& objectsChanged);
    void onPrimAdded(const PXR_NS::SdfPath& path);
    void onPrimRemoved(const PXR_NS::SdfPath& path);
    void onPropertyChanged(const PXR_NS::SdfPath& path);

    void insertAddedPrim(const PXR_NS::SdfPath& primPath, ChangedPrimType primType);
    void insertRemovedPrim(const PXR_NS::SdfPath& primPath, ChangedPrimType primType);
    void insertPropertyChanged(
        const PXR_NS::SdfPath& primPath,
        ChangedPrimType primType,
        const PXR_NS::TfToken& propertyName);

    ChangedPrimType getTypeFromStage(const PXR_NS::SdfPath& path) const;
    ChangedPrimType getTypeFromAssetRegistry(const PXR_NS::SdfPath& path) const;

    Context* _pContext;
    PXR_NS::TfNotice::Key _noticeListenerKey;
    std::vector<ChangedPrim> _changedPrims;
};

} // namespace cesium::omniverse
