#include "cesium/omniverse/OmniImagery.h"

#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/imagery.h>

namespace cesium::omniverse {

OmniImagery::OmniImagery(const pxr::SdfPath& path)
    : _path(path) {}

pxr::SdfPath OmniImagery::getPath() const {
    return _path;
}

std::string OmniImagery::getName() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);
    return imagery.GetPrim().GetName().GetString();
}

bool OmniImagery::getShowCreditsOnScreen() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    bool showCreditsOnScreen;
    imagery.GetShowCreditsOnScreenAttr().Get<bool>(&showCreditsOnScreen);

    return showCreditsOnScreen;
}

double OmniImagery::getAlpha() const {
    auto imagery = UsdUtil::getCesiumImagery(_path);

    float alpha;
    imagery.GetAlphaAttr().Get<float>(&alpha);

    return static_cast<double>(alpha);
}

} // namespace cesium::omniverse
