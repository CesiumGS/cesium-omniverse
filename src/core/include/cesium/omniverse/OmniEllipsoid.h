#pragma once

#include <glm/glm.hpp>
#include <pxr/usd/sdf/path.h>

namespace CesiumGeospatial {
class Ellipsoid;
}

namespace cesium::omniverse {

class Context;

class OmniEllipsoid {
  public:
    OmniEllipsoid(Context* pContext, const pxr::SdfPath& path);
    ~OmniEllipsoid() = default;
    OmniEllipsoid(const OmniEllipsoid&) = delete;
    OmniEllipsoid& operator=(const OmniEllipsoid&) = delete;
    OmniEllipsoid(OmniEllipsoid&&) noexcept = default;
    OmniEllipsoid& operator=(OmniEllipsoid&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] CesiumGeospatial::Ellipsoid getEllipsoid() const;

  private:
    Context* _pContext;
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
