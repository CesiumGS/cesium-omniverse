#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/AccessorView.h>
#include <glm/glm.hpp>

#include <gsl/span>

namespace cesium::omniverse {

class PositionsAccessor {
  public:
    PositionsAccessor();
    PositionsAccessor(const CesiumGltf::AccessorView<glm::fvec3>& view);

    void fill(const gsl::span<glm::fvec3>& values) const;
    [[nodiscard]] const glm::fvec3& get(uint64_t index) const;
    [[nodiscard]] uint64_t size() const;

  private:
    CesiumGltf::AccessorView<glm::fvec3> _view;
    uint64_t _size;
};

class IndicesAccessor {
  public:
    IndicesAccessor();
    IndicesAccessor(uint64_t size);
    IndicesAccessor(const CesiumGltf::AccessorView<uint8_t>& uint8View);
    IndicesAccessor(const CesiumGltf::AccessorView<uint16_t>& uint16View);
    IndicesAccessor(const CesiumGltf::AccessorView<uint32_t>& uint32View);

    template <typename T> static IndicesAccessor FromTriangleStrips(const CesiumGltf::AccessorView<T>& view);
    template <typename T> static IndicesAccessor FromTriangleFans(const CesiumGltf::AccessorView<T>& view);

    void fill(const gsl::span<int>& values) const;
    [[nodiscard]] uint32_t get(uint64_t index) const;
    [[nodiscard]] uint64_t size() const;

  private:
    std::vector<uint32_t> _computed;
    CesiumGltf::AccessorView<uint8_t> _uint8View;
    CesiumGltf::AccessorView<uint16_t> _uint16View;
    CesiumGltf::AccessorView<uint32_t> _uint32View;
    uint64_t _size;
};

class NormalsAccessor {
  public:
    NormalsAccessor();
    NormalsAccessor(const CesiumGltf::AccessorView<glm::fvec3>& view);

    static NormalsAccessor GenerateSmooth(const PositionsAccessor& positions, const IndicesAccessor& indices);

    void fill(const gsl::span<glm::fvec3>& values) const;
    [[nodiscard]] uint64_t size() const;

  private:
    std::vector<glm::fvec3> _computed;
    CesiumGltf::AccessorView<glm::fvec3> _view;
    uint64_t _size;
};

class TexcoordsAccessor {
  public:
    TexcoordsAccessor();
    TexcoordsAccessor(const CesiumGltf::AccessorView<glm::fvec2>& view, bool flipVertical);

    void fill(const gsl::span<glm::fvec2>& values) const;
    [[nodiscard]] uint64_t size() const;

  private:
    CesiumGltf::AccessorView<glm::fvec2> _view;
    bool _flipVertical;
    uint64_t _size;
};

class VertexColorsAccessor {
  public:
    VertexColorsAccessor();
    VertexColorsAccessor(const CesiumGltf::AccessorView<glm::u8vec3>& uint8Vec3View);
    VertexColorsAccessor(const CesiumGltf::AccessorView<glm::u8vec4>& uint8Vec4View);
    VertexColorsAccessor(const CesiumGltf::AccessorView<glm::u16vec3>& uint16Vec3View);
    VertexColorsAccessor(const CesiumGltf::AccessorView<glm::u16vec4>& uint16Vec4View);
    VertexColorsAccessor(const CesiumGltf::AccessorView<glm::fvec3>& float32Vec3View);
    VertexColorsAccessor(const CesiumGltf::AccessorView<glm::fvec4>& float32Vec4View);

    void fill(const gsl::span<glm::fvec3>& values) const;
    [[nodiscard]] uint64_t size() const;

  private:
    CesiumGltf::AccessorView<glm::u8vec3> _uint8Vec3View;
    CesiumGltf::AccessorView<glm::u8vec4> _uint8Vec4View;
    CesiumGltf::AccessorView<glm::u16vec3> _uint16Vec3View;
    CesiumGltf::AccessorView<glm::u16vec4> _uint16Vec4View;
    CesiumGltf::AccessorView<glm::fvec3> _float32Vec3View;
    CesiumGltf::AccessorView<glm::fvec4> _float32Vec4View;

    uint64_t _size;
};

class FaceVertexCountsAccessor {
  public:
    FaceVertexCountsAccessor();
    FaceVertexCountsAccessor(uint64_t size);

    void fill(const gsl::span<int>& values) const;
    [[nodiscard]] uint64_t size() const;

  private:
    uint64_t _size;
};

} // namespace cesium::omniverse
