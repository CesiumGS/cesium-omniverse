#pragma once

#include <glm/fwd.hpp>
#include <pxr/usd/sdf/path.h>

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
