#pragma once

#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumGeospatial/LocalHorizontalCoordinateSystem.h>
#include <pxr/usd/sdf/path.h>

namespace CesiumGeospatial {
class Cartographic;
class LocalHorizontalCoordinateSystem;
} // namespace CesiumGeospatial

namespace cesium::omniverse {

class Context;

class OmniGeoreference {
  public:
    OmniGeoreference(Context* pContext, const pxr::SdfPath& path);
    ~OmniGeoreference() = default;
    OmniGeoreference(const OmniGeoreference&) = delete;
    OmniGeoreference& operator=(const OmniGeoreference&) = delete;
    OmniGeoreference(OmniGeoreference&&) noexcept = default;
    OmniGeoreference& operator=(OmniGeoreference&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] CesiumGeospatial::Cartographic getOrigin() const;
    [[nodiscard]] const CesiumGeospatial::Ellipsoid& getEllipsoid() const;
    [[nodiscard]] CesiumGeospatial::LocalHorizontalCoordinateSystem getLocalCoordinateSystem() const;
    void update();

  private:
    Context* _pContext;
    pxr::SdfPath _path;
    CesiumGeospatial::Ellipsoid _ellipsoid;
};
} // namespace cesium::omniverse
