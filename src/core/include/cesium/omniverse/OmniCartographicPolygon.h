#pragma once

#include <glm/fwd.hpp>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/sdf/path.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace CesiumGeospatial {
class Cartographic;
}

namespace cesium::omniverse {

class Context;

class OmniCartographicPolygon {
  public:
    OmniCartographicPolygon(Context* pContext, const pxr::SdfPath& path);
    ~OmniCartographicPolygon() = default;
    OmniCartographicPolygon(const OmniCartographicPolygon&) = delete;
    OmniCartographicPolygon& operator=(const OmniCartographicPolygon&) = delete;
    OmniCartographicPolygon(OmniCartographicPolygon&&) noexcept = default;
    OmniCartographicPolygon& operator=(OmniCartographicPolygon&&) noexcept = default;

    [[nodiscard]] const pxr::SdfPath& getPath() const;
    [[nodiscard]] std::vector<CesiumGeospatial::Cartographic> getCartographics() const;

  private:
    Context* _pContext;
    pxr::SdfPath _path;
};
} // namespace cesium::omniverse
