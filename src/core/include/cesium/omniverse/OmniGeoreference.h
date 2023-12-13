#pragma once

#include <CesiumGeospatial/Cartographic.h>
#include <pxr/usd/sdf/path.h>

namespace cesium::omniverse {
class OmniGeoreference {
  public:
    OmniGeoreference(const pxr::SdfPath& path);

    [[nodiscard]] pxr::SdfPath getPath() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getOrigin() const;
    void setOrigin(const CesiumGeospatial::Cartographic& origin) const;

  private:
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
