#pragma once

#include "cesium/omniverse/VertexAttributeType.h"

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

    /**
     * @brief Copy accessor values to the given output values, including any data transformations.
     *
     * @param values The output values.
     * @param repeat Indicates how many times each value in the accessor should be repeated in the output. Typically repeat is 1, but for voxel point clouds repeat is 8.
     */
    void fill(const gsl::span<glm::fvec4>& values, uint64_t repeat = 1) const;

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

class VertexIdsAccessor {
  public:
    VertexIdsAccessor();
    VertexIdsAccessor(uint64_t size);

    void fill(const gsl::span<float>& values, uint64_t repeat = 1) const;
    [[nodiscard]] uint64_t size() const;

  private:
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

template <VertexAttributeType T> class VertexAttributeAccessor {
  public:
    VertexAttributeAccessor()
        : _size(0){};
    VertexAttributeAccessor(const CesiumGltf::AccessorView<GetRawType<T>>& view)
        : _view(view)
        , _size(static_cast<uint64_t>(view.size())) {}

    void fill(const gsl::span<GetPrimvarType<T>>& values, uint64_t repeat = 1) const {
        const auto size = values.size();
        assert(size == _size * repeat);

        for (uint64_t i = 0; i < size; i++) {
            values[i] = static_cast<GetPrimvarType<T>>(_view[static_cast<int64_t>(i / repeat)]);
        }
    }

    [[nodiscard]] uint64_t size() const {
        return _size;
    }

  private:
    CesiumGltf::AccessorView<GetRawType<T>> _view;
    uint64_t _size;
};

} // namespace cesium::omniverse
