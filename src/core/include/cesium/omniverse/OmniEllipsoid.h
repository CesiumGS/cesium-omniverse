#pragma once

#include <glm/glm.hpp>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/sdf/path.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

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
