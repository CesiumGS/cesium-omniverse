#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniGeoreference {
  public:
    OmniGeoreference(const pxr::SdfPath& path);

    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getCartographic() const;
    void setCartographic(const CesiumGeospatial::Cartographic& cartographic) const;

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
