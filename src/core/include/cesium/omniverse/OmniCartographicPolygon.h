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
    OmniCartographicPolygon(Context* pContext, const PXR_NS::SdfPath& path);
    ~OmniCartographicPolygon() = default;
    OmniCartographicPolygon(const OmniCartographicPolygon&) = delete;
    OmniCartographicPolygon& operator=(const OmniCartographicPolygon&) = delete;
    OmniCartographicPolygon(OmniCartographicPolygon&&) noexcept = default;
    OmniCartographicPolygon& operator=(OmniCartographicPolygon&&) noexcept = default;

    [[nodiscard]] const PXR_NS::SdfPath& getPath() const;
    [[nodiscard]] std::vector<CesiumGeospatial::Cartographic> getCartographics() const;

  private:
    Context* _pContext;
    PXR_NS::SdfPath _path;
};
} // namespace cesium::omniverse
