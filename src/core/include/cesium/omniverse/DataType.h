#pragma once

#include <glm/glm.hpp>
#include <omni/fabric/Type.h>

#include <array>

// GLM does not typedef all integer matrix types, so they are typedef'd here
namespace glm {
using u8mat2 = mat<2, 2, uint8_t>;
using i8mat2 = mat<2, 2, int8_t>;
using u16mat2 = mat<2, 2, uint16_t>;
using i16mat2 = mat<2, 2, int16_t>;
using u32mat2 = mat<2, 2, uint32_t>;
using i32mat2 = mat<2, 2, int32_t>;
using u64mat2 = mat<2, 2, uint64_t>;
using i64mat2 = mat<2, 2, int64_t>;
using u8mat3 = mat<3, 3, uint8_t>;
using i8mat3 = mat<3, 3, int8_t>;
using u16mat3 = mat<3, 3, uint16_t>;
using i16mat3 = mat<3, 3, int16_t>;
using u32mat3 = mat<3, 3, uint32_t>;
using i32mat3 = mat<3, 3, int32_t>;
using u64mat3 = mat<3, 3, uint64_t>;
using i64mat3 = mat<3, 3, int64_t>;
using u8mat4 = mat<4, 4, uint8_t>;
using i8mat4 = mat<4, 4, int8_t>;
using u16mat4 = mat<4, 4, uint16_t>;
using i16mat4 = mat<4, 4, int16_t>;
using u32mat4 = mat<4, 4, uint32_t>;
using i32mat4 = mat<4, 4, int32_t>;
using u64mat4 = mat<4, 4, uint64_t>;
using i64mat4 = mat<4, 4, int64_t>;
} // namespace glm

namespace cesium::omniverse {

enum class DataType {
    UINT8,
    INT8,
    UINT16,
    INT16,
    UINT32,
    INT32,
    UINT64,
    INT64,
    FLOAT32,
    FLOAT64,
    UINT8_NORM,
    INT8_NORM,
    UINT16_NORM,
    INT16_NORM,
    UINT32_NORM,
    INT32_NORM,
    UINT64_NORM,
    INT64_NORM,
    VEC2_UINT8,
    VEC2_INT8,
    VEC2_UINT16,
    VEC2_INT16,
    VEC2_UINT32,
    VEC2_INT32,
    VEC2_UINT64,
    VEC2_INT64,
    VEC2_FLOAT32,
    VEC2_FLOAT64,
    VEC2_UINT8_NORM,
    VEC2_INT8_NORM,
    VEC2_UINT16_NORM,
    VEC2_INT16_NORM,
    VEC2_UINT32_NORM,
    VEC2_INT32_NORM,
    VEC2_UINT64_NORM,
    VEC2_INT64_NORM,
    VEC3_UINT8,
    VEC3_INT8,
    VEC3_UINT16,
    VEC3_INT16,
    VEC3_UINT32,
    VEC3_INT32,
    VEC3_UINT64,
    VEC3_INT64,
    VEC3_FLOAT32,
    VEC3_FLOAT64,
    VEC3_UINT8_NORM,
    VEC3_INT8_NORM,
    VEC3_UINT16_NORM,
    VEC3_INT16_NORM,
    VEC3_UINT32_NORM,
    VEC3_INT32_NORM,
    VEC3_UINT64_NORM,
    VEC3_INT64_NORM,
    VEC4_UINT8,
    VEC4_INT8,
    VEC4_UINT16,
    VEC4_INT16,
    VEC4_UINT32,
    VEC4_INT32,
    VEC4_UINT64,
    VEC4_INT64,
    VEC4_FLOAT32,
    VEC4_FLOAT64,
    VEC4_UINT8_NORM,
    VEC4_INT8_NORM,
    VEC4_UINT16_NORM,
    VEC4_INT16_NORM,
    VEC4_UINT32_NORM,
    VEC4_INT32_NORM,
    VEC4_UINT64_NORM,
    VEC4_INT64_NORM,
    MAT2_UINT8,
    MAT2_INT8,
    MAT2_UINT16,
    MAT2_INT16,
    MAT2_UINT32,
    MAT2_INT32,
    MAT2_UINT64,
    MAT2_INT64,
    MAT2_FLOAT32,
    MAT2_FLOAT64,
    MAT2_UINT8_NORM,
    MAT2_INT8_NORM,
    MAT2_UINT16_NORM,
    MAT2_INT16_NORM,
    MAT2_UINT32_NORM,
    MAT2_INT32_NORM,
    MAT2_UINT64_NORM,
    MAT2_INT64_NORM,
    MAT3_UINT8,
    MAT3_INT8,
    MAT3_UINT16,
    MAT3_INT16,
    MAT3_UINT32,
    MAT3_INT32,
    MAT3_UINT64,
    MAT3_INT64,
    MAT3_FLOAT32,
    MAT3_FLOAT64,
    MAT3_UINT8_NORM,
    MAT3_INT8_NORM,
    MAT3_UINT16_NORM,
    MAT3_INT16_NORM,
    MAT3_UINT32_NORM,
    MAT3_INT32_NORM,
    MAT3_UINT64_NORM,
    MAT3_INT64_NORM,
    MAT4_UINT8,
    MAT4_INT8,
    MAT4_UINT16,
    MAT4_INT16,
    MAT4_UINT32,
    MAT4_INT32,
    MAT4_UINT64,
    MAT4_INT64,
    MAT4_FLOAT32,
    MAT4_FLOAT64,
    MAT4_UINT8_NORM,
    MAT4_INT8_NORM,
    MAT4_UINT16_NORM,
    MAT4_INT16_NORM,
    MAT4_UINT32_NORM,
    MAT4_INT32_NORM,
    MAT4_UINT64_NORM,
    MAT4_INT64_NORM,
};

constexpr auto DataTypeCount = static_cast<uint64_t>(DataType::MAT4_INT64_NORM) + 1;

enum class MdlInternalPropertyType {
    INT32,
    FLOAT32,
    INT32_NORM,
    VEC2_INT32,
    VEC2_FLOAT32,
    VEC2_INT32_NORM,
    VEC3_INT32,
    VEC3_FLOAT32,
    VEC3_INT32_NORM,
    VEC4_INT32,
    VEC4_FLOAT32,
    VEC4_INT32_NORM,
    MAT2_INT32,
    MAT2_FLOAT32,
    MAT2_INT32_NORM,
    MAT3_INT32,
    MAT3_FLOAT32,
    MAT3_INT32_NORM,
    MAT4_INT32,
    MAT4_FLOAT32,
    MAT4_INT32_NORM,
};

constexpr auto MdlInternalPropertyTypeCount = static_cast<uint64_t>(MdlInternalPropertyType::MAT4_INT32_NORM) + 1;

enum class MdlExternalPropertyType {
    INT32,
    FLOAT32,
    VEC2_INT32,
    VEC2_FLOAT32,
    VEC3_INT32,
    VEC3_FLOAT32,
    VEC4_INT32,
    VEC4_FLOAT32,
    MAT2_FLOAT32,
    MAT3_FLOAT32,
    MAT4_FLOAT32,
};

constexpr auto MdlExternalPropertyTypeCount = static_cast<uint64_t>(MdlExternalPropertyType::MAT4_FLOAT32) + 1;

template <DataType T> struct IsNormalizedImpl;
template <> struct IsNormalizedImpl<DataType::UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::INT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC2_INT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC3_INT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::VEC4_INT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT2_INT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT3_INT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT8> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT16> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_FLOAT32> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_FLOAT64> : std::false_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT8_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT16_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT32_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_UINT64_NORM> : std::true_type {};
template <> struct IsNormalizedImpl<DataType::MAT4_INT64_NORM> : std::true_type {};

template <DataType T> struct IsFloatingPointImpl;
template <> struct IsFloatingPointImpl<DataType::UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::INT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC2_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC3_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::VEC4_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT2_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT3_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT8> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT16> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT32> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT64> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_FLOAT32> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_FLOAT64> : std::true_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPointImpl<DataType::MAT4_INT64_NORM> : std::false_type {};

template <DataType T> struct IsMatrixImpl;
template <> struct IsMatrixImpl<DataType::UINT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::FLOAT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::FLOAT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::UINT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::INT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_FLOAT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_FLOAT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_UINT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC2_INT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_FLOAT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_FLOAT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_UINT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC3_INT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT8> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT16> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_FLOAT32> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_FLOAT64> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT8_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT16_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT32_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_UINT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::VEC4_INT64_NORM> : std::false_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT8> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT8> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT16> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT16> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_FLOAT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_FLOAT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT8_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT8_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT16_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT16_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT32_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT32_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_UINT64_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT2_INT64_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT8> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT8> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT16> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT16> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_FLOAT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_FLOAT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT8_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT8_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT16_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT16_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT32_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT32_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_UINT64_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT3_INT64_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT8> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT8> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT16> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT16> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_FLOAT32> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_FLOAT64> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT8_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT8_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT16_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT16_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT32_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT32_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_UINT64_NORM> : std::true_type {};
template <> struct IsMatrixImpl<DataType::MAT4_INT64_NORM> : std::true_type {};

template <DataType T> struct GetRawTypeImpl;
template <> struct GetRawTypeImpl<DataType::UINT8> { using Type = uint8_t; };
template <> struct GetRawTypeImpl<DataType::INT8> { using Type = int8_t; };
template <> struct GetRawTypeImpl<DataType::UINT16> { using Type = uint16_t; };
template <> struct GetRawTypeImpl<DataType::INT16> { using Type = int16_t; };
template <> struct GetRawTypeImpl<DataType::UINT32> { using Type = uint32_t; };
template <> struct GetRawTypeImpl<DataType::INT32> { using Type = int32_t; };
template <> struct GetRawTypeImpl<DataType::UINT64> { using Type = uint64_t; };
template <> struct GetRawTypeImpl<DataType::INT64> { using Type = int64_t; };
template <> struct GetRawTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetRawTypeImpl<DataType::FLOAT64> { using Type = double; };
template <> struct GetRawTypeImpl<DataType::UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawTypeImpl<DataType::INT8_NORM> { using Type = int8_t; };
template <> struct GetRawTypeImpl<DataType::UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawTypeImpl<DataType::INT16_NORM> { using Type = int16_t; };
template <> struct GetRawTypeImpl<DataType::UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawTypeImpl<DataType::INT32_NORM> { using Type = int32_t; };
template <> struct GetRawTypeImpl<DataType::UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawTypeImpl<DataType::INT64_NORM> { using Type = int64_t; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT8> { using Type = glm::u8vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT8> { using Type = glm::i8vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT16> { using Type = glm::u16vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT16> { using Type = glm::i16vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT32> { using Type = glm::u32vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT64> { using Type = glm::u64vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT64> { using Type = glm::i64vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_FLOAT64> { using Type = glm::f64vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::u8vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::i8vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::u16vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::i16vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = glm::u32vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT32_NORM> { using Type = glm::i32vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = glm::u64vec2; };
template <> struct GetRawTypeImpl<DataType::VEC2_INT64_NORM> { using Type = glm::i64vec2; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT8> { using Type = glm::u8vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT8> { using Type = glm::i8vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT16> { using Type = glm::u16vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT16> { using Type = glm::i16vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT32> { using Type = glm::u32vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT64> { using Type = glm::u64vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT64> { using Type = glm::i64vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_FLOAT64> { using Type = glm::f64vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::u8vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::i8vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::u16vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::i16vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = glm::u32vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT32_NORM> { using Type = glm::i32vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = glm::u64vec3; };
template <> struct GetRawTypeImpl<DataType::VEC3_INT64_NORM> { using Type = glm::i64vec3; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT8> { using Type = glm::u8vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT8> { using Type = glm::i8vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT16> { using Type = glm::u16vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT16> { using Type = glm::i16vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT32> { using Type = glm::u32vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT64> { using Type = glm::u64vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT64> { using Type = glm::i64vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_FLOAT64> { using Type = glm::f64vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::u8vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::i8vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::u16vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::i16vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = glm::u32vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT32_NORM> { using Type = glm::i32vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = glm::u64vec4; };
template <> struct GetRawTypeImpl<DataType::VEC4_INT64_NORM> { using Type = glm::i64vec4; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT8> { using Type = glm::u8mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT8> { using Type = glm::i8mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT16> { using Type = glm::u16mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT16> { using Type = glm::i16mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT32> { using Type = glm::u32mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT32> { using Type = glm::i32mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT64> { using Type = glm::u64mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT64> { using Type = glm::i64mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f64mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::u8mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::i8mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::u16mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::i16mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::u32mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::i32mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::u64mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::i64mat2; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT8> { using Type = glm::u8mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT8> { using Type = glm::i8mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT16> { using Type = glm::u16mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT16> { using Type = glm::i16mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT32> { using Type = glm::u32mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT32> { using Type = glm::i32mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT64> { using Type = glm::u64mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT64> { using Type = glm::i64mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f64mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::u8mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::i8mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::u16mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::i16mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::u32mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::i32mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::u64mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::i64mat3; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT8> { using Type = glm::u8mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT8> { using Type = glm::i8mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT16> { using Type = glm::u16mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT16> { using Type = glm::i16mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT32> { using Type = glm::u32mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT32> { using Type = glm::i32mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT64> { using Type = glm::u64mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT64> { using Type = glm::i64mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f64mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::u8mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::i8mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::u16mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::i16mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::u32mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::i32mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::u64mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::i64mat4; };
template <DataType T> using GetRawType = typename GetRawTypeImpl<T>::Type;

template <DataType T> struct GetRawComponentTypeImpl;
template <> struct GetRawComponentTypeImpl<DataType::UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::INT64_NORM> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC2_INT64_NORM> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC3_INT64_NORM> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::VEC4_INT64_NORM> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT2_INT64_NORM> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT3_INT64_NORM> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT8> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT8> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT16> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT16> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT32> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT32> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT64> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT64> { using Type = int64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_FLOAT32> { using Type = float; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_FLOAT64> { using Type = double; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT8_NORM> { using Type = int8_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT16_NORM> { using Type = int16_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = uint32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT32_NORM> { using Type = int32_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = uint64_t; };
template <> struct GetRawComponentTypeImpl<DataType::MAT4_INT64_NORM> { using Type = int64_t; };
template <DataType T> using GetRawComponentType = typename GetRawComponentTypeImpl<T>::Type;

template <DataType T> struct GetTransformedTypeImpl;
template <> struct GetTransformedTypeImpl<DataType::UINT8> { using Type = uint8_t; };
template <> struct GetTransformedTypeImpl<DataType::INT8> { using Type = int8_t; };
template <> struct GetTransformedTypeImpl<DataType::UINT16> { using Type = uint16_t; };
template <> struct GetTransformedTypeImpl<DataType::INT16> { using Type = int16_t; };
template <> struct GetTransformedTypeImpl<DataType::UINT32> { using Type = uint32_t; };
template <> struct GetTransformedTypeImpl<DataType::INT32> { using Type = int32_t; };
template <> struct GetTransformedTypeImpl<DataType::UINT64> { using Type = uint64_t; };
template <> struct GetTransformedTypeImpl<DataType::INT64> { using Type = int64_t; };
template <> struct GetTransformedTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetTransformedTypeImpl<DataType::FLOAT64> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::UINT8_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::INT8_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::UINT16_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::INT16_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::UINT32_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::INT32_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::UINT64_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::INT64_NORM> { using Type = double; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT8> { using Type = glm::u8vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT8> { using Type = glm::i8vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT16> { using Type = glm::u16vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT16> { using Type = glm::i16vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT32> { using Type = glm::u32vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT64> { using Type = glm::u64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT64> { using Type = glm::i64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_FLOAT64> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT32_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT64_NORM> { using Type = glm::f64vec2; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT8> { using Type = glm::u8vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT8> { using Type = glm::i8vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT16> { using Type = glm::u16vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT16> { using Type = glm::i16vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT32> { using Type = glm::u32vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT64> { using Type = glm::u64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT64> { using Type = glm::i64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_FLOAT64> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT32_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT64_NORM> { using Type = glm::f64vec3; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT8> { using Type = glm::u8vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT8> { using Type = glm::i8vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT16> { using Type = glm::u16vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT16> { using Type = glm::i16vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT32> { using Type = glm::u32vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT64> { using Type = glm::u64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT64> { using Type = glm::i64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_FLOAT64> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT32_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT64_NORM> { using Type = glm::f64vec4; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT8> { using Type = glm::u8mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT8> { using Type = glm::i8mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT16> { using Type = glm::u16mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT16> { using Type = glm::i16mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT32> { using Type = glm::u32mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT32> { using Type = glm::i32mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT64> { using Type = glm::u64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT64> { using Type = glm::i64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::f64mat2; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT8> { using Type = glm::u8mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT8> { using Type = glm::i8mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT16> { using Type = glm::u16mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT16> { using Type = glm::i16mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT32> { using Type = glm::u32mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT32> { using Type = glm::i32mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT64> { using Type = glm::u64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT64> { using Type = glm::i64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::f64mat3; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT8> { using Type = glm::u8mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT8> { using Type = glm::i8mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT16> { using Type = glm::u16mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT16> { using Type = glm::i16mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT32> { using Type = glm::u32mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT32> { using Type = glm::i32mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT64> { using Type = glm::u64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT64> { using Type = glm::i64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::f64mat4; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::f64mat4; };
template <DataType T> using GetTransformedType = typename GetTransformedTypeImpl<T>::Type;

// clang-format off
template <MdlInternalPropertyType T> struct GetMdlInternalPropertyRawTypeImpl;
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::INT32> { using Type = int; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::FLOAT32> { using Type = float; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::INT32_NORM> { using Type = int; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC2_INT32_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC3_INT32_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::VEC4_INT32_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT2_INT32> { using Type = glm::i32mat2; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT2_INT32_NORM> { using Type = glm::i32mat2; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT3_INT32> { using Type = glm::i32mat3; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT3_INT32_NORM> { using Type = glm::i32mat3; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT4_INT32> { using Type = glm::i32mat4; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetMdlInternalPropertyRawTypeImpl<MdlInternalPropertyType::MAT4_INT32_NORM> { using Type = glm::i32mat4; };
template <MdlInternalPropertyType T> using GetMdlInternalPropertyRawType = typename GetMdlInternalPropertyRawTypeImpl<T>::Type;
// clang-format on

// clang-format off
template <MdlInternalPropertyType T> struct GetMdlInternalPropertyTransformedTypeImplImpl;
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::INT32> { using Type = int; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::FLOAT32> { using Type = float; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::INT32_NORM> { using Type = float; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC2_INT32_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC3_INT32_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::VEC4_INT32_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT2_INT32> { using Type = glm::i32mat2; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT2_INT32_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT3_INT32> { using Type = glm::i32mat3; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT3_INT32_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT4_INT32> { using Type = glm::i32mat4; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetMdlInternalPropertyTransformedTypeImplImpl<MdlInternalPropertyType::MAT4_INT32_NORM> { using Type = glm::f32mat4; };
template <MdlInternalPropertyType T> using GetMdlInternalPropertyTransformedType = typename GetMdlInternalPropertyTransformedTypeImplImpl<T>::Type;
// clang-format on

// clang-format off
template <DataType T> struct GetMdlInternalPropertyTypeImpl;
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT8> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT8> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT16> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT16> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT32> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT32> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT64> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT64> { static constexpr auto Type = MdlInternalPropertyType::INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT8> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT8> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT16> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT16> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT32> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT32> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT64> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT64> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::VEC2_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::VEC2_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT8> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT8> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT16> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT16> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT32> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT32> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT64> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT64> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::VEC3_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::VEC3_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT8> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT8> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT16> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT16> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT32> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT32> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT64> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT64> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::VEC4_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::VEC4_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::VEC4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT8> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT8> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT16> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT16> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT32> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT32> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT64> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT64> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::MAT2_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::MAT2_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT2_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT8> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT8> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT16> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT16> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT32> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT32> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT64> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT64> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::MAT3_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::MAT3_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT3_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT8> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT8> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT16> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT16> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT32> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT32> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT64> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT64> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto Type = MdlInternalPropertyType::MAT4_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto Type = MdlInternalPropertyType::MAT4_FLOAT32;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
template <> struct GetMdlInternalPropertyTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto Type = MdlInternalPropertyType::MAT4_INT32_NORM;};
// clang-format on

template <DataType T> struct GetComponentCountImpl;
template <> struct GetComponentCountImpl<DataType::UINT8> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT8> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT16> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT16> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT32> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT32> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT64> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT64> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::FLOAT32> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::FLOAT64> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT8_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT8_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT16_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT16_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT32_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT32_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::UINT64_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::INT64_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT8> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT8> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT16> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT16> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT32> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT32> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT64> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT64> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_FLOAT32> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_FLOAT64> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT8_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT16_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT32_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC2_INT64_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT8> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT8> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT16> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT16> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT32> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT32> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT64> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT64> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_FLOAT32> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_FLOAT64> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT8_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT16_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT32_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC3_INT64_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT8> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT8> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT16> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT16> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT64> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT64> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_FLOAT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_FLOAT64> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT8_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT16_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT32_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::VEC4_INT64_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT8> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT8> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT16> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT16> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT64> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT64> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_FLOAT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_FLOAT64> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT8_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT16_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT32_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT2_INT64_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT8> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT8> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT16> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT16> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT32> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT32> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT64> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT64> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_FLOAT32> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_FLOAT64> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT8_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT16_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT32_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT3_INT64_NORM> { static constexpr auto ComponentCount = 9; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT8> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT8> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT16> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT16> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT32> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT32> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT64> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT64> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_FLOAT32> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_FLOAT64> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT8_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT16_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT32_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto ComponentCount = 16; };
template <> struct GetComponentCountImpl<DataType::MAT4_INT64_NORM> { static constexpr auto ComponentCount = 16; };

// clang-format off
template <typename RawType, typename TransformedType> struct GetTypeReverseImpl;
template <> struct GetTypeReverseImpl<uint8_t, uint8_t> { static constexpr auto Type = DataType::UINT8; };
template <> struct GetTypeReverseImpl<int8_t, int8_t> { static constexpr auto Type = DataType::INT8; };
template <> struct GetTypeReverseImpl<uint16_t, uint16_t> { static constexpr auto Type = DataType::UINT16; };
template <> struct GetTypeReverseImpl<int16_t, int16_t> { static constexpr auto Type = DataType::INT16; };
template <> struct GetTypeReverseImpl<uint32_t, uint32_t> { static constexpr auto Type = DataType::UINT32; };
template <> struct GetTypeReverseImpl<int32_t, int32_t> { static constexpr auto Type = DataType::INT32; };
template <> struct GetTypeReverseImpl<uint64_t, uint64_t> { static constexpr auto Type = DataType::UINT64; };
template <> struct GetTypeReverseImpl<int64_t, int64_t> { static constexpr auto Type = DataType::INT64; };
template <> struct GetTypeReverseImpl<float, float> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetTypeReverseImpl<double, double> { static constexpr auto Type = DataType::FLOAT64; };
template <> struct GetTypeReverseImpl<glm::u8vec2, glm::u8vec2> { static constexpr auto Type = DataType::VEC2_UINT8; };
template <> struct GetTypeReverseImpl<glm::i8vec2, glm::i8vec2> { static constexpr auto Type = DataType::VEC2_INT8; };
template <> struct GetTypeReverseImpl<glm::u16vec2, glm::u16vec2> { static constexpr auto Type = DataType::VEC2_UINT16; };
template <> struct GetTypeReverseImpl<glm::i16vec2, glm::i16vec2> { static constexpr auto Type = DataType::VEC2_INT16; };
template <> struct GetTypeReverseImpl<glm::u32vec2, glm::u32vec2> { static constexpr auto Type = DataType::VEC2_UINT32; };
template <> struct GetTypeReverseImpl<glm::i32vec2, glm::i32vec2> { static constexpr auto Type = DataType::VEC2_INT32; };
template <> struct GetTypeReverseImpl<glm::u64vec2, glm::u64vec2> { static constexpr auto Type = DataType::VEC2_UINT64; };
template <> struct GetTypeReverseImpl<glm::i64vec2, glm::i64vec2> { static constexpr auto Type = DataType::VEC2_INT64; };
template <> struct GetTypeReverseImpl<glm::f32vec2, glm::f32vec2> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetTypeReverseImpl<glm::f64vec2, glm::f64vec2> { static constexpr auto Type = DataType::VEC2_FLOAT64; };
template <> struct GetTypeReverseImpl<glm::u8vec3, glm::u8vec3> { static constexpr auto Type = DataType::VEC3_UINT8; };
template <> struct GetTypeReverseImpl<glm::i8vec3, glm::i8vec3> { static constexpr auto Type = DataType::VEC3_INT8; };
template <> struct GetTypeReverseImpl<glm::u16vec3, glm::u16vec3> { static constexpr auto Type = DataType::VEC3_UINT16; };
template <> struct GetTypeReverseImpl<glm::i16vec3, glm::i16vec3> { static constexpr auto Type = DataType::VEC3_INT16; };
template <> struct GetTypeReverseImpl<glm::u32vec3, glm::u32vec3> { static constexpr auto Type = DataType::VEC3_UINT32; };
template <> struct GetTypeReverseImpl<glm::i32vec3, glm::i32vec3> { static constexpr auto Type = DataType::VEC3_INT32; };
template <> struct GetTypeReverseImpl<glm::u64vec3, glm::u64vec3> { static constexpr auto Type = DataType::VEC3_UINT64; };
template <> struct GetTypeReverseImpl<glm::i64vec3, glm::i64vec3> { static constexpr auto Type = DataType::VEC3_INT64; };
template <> struct GetTypeReverseImpl<glm::f32vec3, glm::f32vec3> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetTypeReverseImpl<glm::f64vec3, glm::f64vec3> { static constexpr auto Type = DataType::VEC3_FLOAT64; };
template <> struct GetTypeReverseImpl<glm::u8vec4, glm::u8vec4> { static constexpr auto Type = DataType::VEC4_UINT8; };
template <> struct GetTypeReverseImpl<glm::i8vec4, glm::i8vec4> { static constexpr auto Type = DataType::VEC4_INT8; };
template <> struct GetTypeReverseImpl<glm::u16vec4, glm::u16vec4> { static constexpr auto Type = DataType::VEC4_UINT16; };
template <> struct GetTypeReverseImpl<glm::i16vec4, glm::i16vec4> { static constexpr auto Type = DataType::VEC4_INT16; };
template <> struct GetTypeReverseImpl<glm::u32vec4, glm::u32vec4> { static constexpr auto Type = DataType::VEC4_UINT32; };
template <> struct GetTypeReverseImpl<glm::i32vec4, glm::i32vec4> { static constexpr auto Type = DataType::VEC4_INT32; };
template <> struct GetTypeReverseImpl<glm::u64vec4, glm::u64vec4> { static constexpr auto Type = DataType::VEC4_UINT64; };
template <> struct GetTypeReverseImpl<glm::i64vec4, glm::i64vec4> { static constexpr auto Type = DataType::VEC4_INT64; };
template <> struct GetTypeReverseImpl<glm::f32vec4, glm::f32vec4> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetTypeReverseImpl<glm::f64vec4, glm::f64vec4> { static constexpr auto Type = DataType::VEC4_FLOAT64; };
template <> struct GetTypeReverseImpl<glm::u8mat2, glm::u8mat2> { static constexpr auto Type = DataType::MAT2_UINT8; };
template <> struct GetTypeReverseImpl<glm::i8mat2, glm::i8mat2> { static constexpr auto Type = DataType::MAT2_INT8; };
template <> struct GetTypeReverseImpl<glm::u16mat2, glm::u16mat2> { static constexpr auto Type = DataType::MAT2_UINT16; };
template <> struct GetTypeReverseImpl<glm::i16mat2, glm::i16mat2> { static constexpr auto Type = DataType::MAT2_INT16; };
template <> struct GetTypeReverseImpl<glm::u32mat2, glm::u32mat2> { static constexpr auto Type = DataType::MAT2_UINT32; };
template <> struct GetTypeReverseImpl<glm::i32mat2, glm::i32mat2> { static constexpr auto Type = DataType::MAT2_INT32; };
template <> struct GetTypeReverseImpl<glm::u64mat2, glm::u64mat2> { static constexpr auto Type = DataType::MAT2_UINT64; };
template <> struct GetTypeReverseImpl<glm::i64mat2, glm::i64mat2> { static constexpr auto Type = DataType::MAT2_INT64; };
template <> struct GetTypeReverseImpl<glm::f32mat2, glm::f32mat2> { static constexpr auto Type = DataType::MAT2_FLOAT32; };
template <> struct GetTypeReverseImpl<glm::f64mat2, glm::f32mat2> { static constexpr auto Type = DataType::MAT2_FLOAT64; };
template <> struct GetTypeReverseImpl<glm::u8mat3, glm::u8mat3> { static constexpr auto Type = DataType::MAT3_UINT8; };
template <> struct GetTypeReverseImpl<glm::i8mat3, glm::i8mat3> { static constexpr auto Type = DataType::MAT3_INT8; };
template <> struct GetTypeReverseImpl<glm::u16mat3, glm::u16mat3> { static constexpr auto Type = DataType::MAT3_UINT16; };
template <> struct GetTypeReverseImpl<glm::i16mat3, glm::i16mat3> { static constexpr auto Type = DataType::MAT3_INT16; };
template <> struct GetTypeReverseImpl<glm::u32mat3, glm::u32mat3> { static constexpr auto Type = DataType::MAT3_UINT32; };
template <> struct GetTypeReverseImpl<glm::i32mat3, glm::i32mat3> { static constexpr auto Type = DataType::MAT3_INT32; };
template <> struct GetTypeReverseImpl<glm::u64mat3, glm::u64mat3> { static constexpr auto Type = DataType::MAT3_UINT64; };
template <> struct GetTypeReverseImpl<glm::i64mat3, glm::i64mat3> { static constexpr auto Type = DataType::MAT3_INT64; };
template <> struct GetTypeReverseImpl<glm::f32mat3, glm::f32mat3> { static constexpr auto Type = DataType::MAT3_FLOAT32; };
template <> struct GetTypeReverseImpl<glm::f64mat3, glm::f64mat3> { static constexpr auto Type = DataType::MAT3_FLOAT64; };
template <> struct GetTypeReverseImpl<glm::u8mat4, glm::u8mat4> { static constexpr auto Type = DataType::MAT4_UINT8; };
template <> struct GetTypeReverseImpl<glm::i8mat4, glm::i8mat4> { static constexpr auto Type = DataType::MAT4_INT8; };
template <> struct GetTypeReverseImpl<glm::u16mat4, glm::u16mat4> { static constexpr auto Type = DataType::MAT4_UINT16; };
template <> struct GetTypeReverseImpl<glm::i16mat4, glm::i16mat4> { static constexpr auto Type = DataType::MAT4_INT16; };
template <> struct GetTypeReverseImpl<glm::u32mat4, glm::u32mat4> { static constexpr auto Type = DataType::MAT4_UINT32; };
template <> struct GetTypeReverseImpl<glm::i32mat4, glm::i32mat4> { static constexpr auto Type = DataType::MAT4_INT32; };
template <> struct GetTypeReverseImpl<glm::u64mat4, glm::u64mat4> { static constexpr auto Type = DataType::MAT4_UINT64; };
template <> struct GetTypeReverseImpl<glm::i64mat4, glm::i64mat4> { static constexpr auto Type = DataType::MAT4_INT64; };
template <> struct GetTypeReverseImpl<glm::f32mat4, glm::f32mat4> { static constexpr auto Type = DataType::MAT4_FLOAT32; };
template <> struct GetTypeReverseImpl<glm::f64mat4, glm::f64mat4> { static constexpr auto Type = DataType::MAT4_FLOAT64; };
template <> struct GetTypeReverseImpl<uint8_t, double> { static constexpr auto Type = DataType::UINT8_NORM; };
template <> struct GetTypeReverseImpl<int8_t, double> { static constexpr auto Type = DataType::INT8_NORM; };
template <> struct GetTypeReverseImpl<uint16_t, double> { static constexpr auto Type = DataType::UINT16_NORM; };
template <> struct GetTypeReverseImpl<int16_t, double> { static constexpr auto Type = DataType::INT16_NORM; };
template <> struct GetTypeReverseImpl<uint32_t, double> { static constexpr auto Type = DataType::UINT32_NORM; };
template <> struct GetTypeReverseImpl<int32_t, double> { static constexpr auto Type = DataType::INT32_NORM; };
template <> struct GetTypeReverseImpl<uint64_t, double> { static constexpr auto Type = DataType::UINT64_NORM; };
template <> struct GetTypeReverseImpl<int64_t, double> { static constexpr auto Type = DataType::INT64_NORM; };
template <> struct GetTypeReverseImpl<glm::u8vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT8_NORM; };
template <> struct GetTypeReverseImpl<glm::i8vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT8_NORM; };
template <> struct GetTypeReverseImpl<glm::u16vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT16_NORM; };
template <> struct GetTypeReverseImpl<glm::i16vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT16_NORM; };
template <> struct GetTypeReverseImpl<glm::u32vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT32_NORM; };
template <> struct GetTypeReverseImpl<glm::i32vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT32_NORM; };
template <> struct GetTypeReverseImpl<glm::u64vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT64_NORM; };
template <> struct GetTypeReverseImpl<glm::i64vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT64_NORM; };
template <> struct GetTypeReverseImpl<glm::u8vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT8_NORM; };
template <> struct GetTypeReverseImpl<glm::i8vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT8_NORM; };
template <> struct GetTypeReverseImpl<glm::u16vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT16_NORM; };
template <> struct GetTypeReverseImpl<glm::i16vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT16_NORM; };
template <> struct GetTypeReverseImpl<glm::u32vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT32_NORM; };
template <> struct GetTypeReverseImpl<glm::i32vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT32_NORM; };
template <> struct GetTypeReverseImpl<glm::u64vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT64_NORM; };
template <> struct GetTypeReverseImpl<glm::i64vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT64_NORM; };
template <> struct GetTypeReverseImpl<glm::u8vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT8_NORM; };
template <> struct GetTypeReverseImpl<glm::i8vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT8_NORM; };
template <> struct GetTypeReverseImpl<glm::u16vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT16_NORM; };
template <> struct GetTypeReverseImpl<glm::i16vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT16_NORM; };
template <> struct GetTypeReverseImpl<glm::u32vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT32_NORM; };
template <> struct GetTypeReverseImpl<glm::i32vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT32_NORM; };
template <> struct GetTypeReverseImpl<glm::u64vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT64_NORM; };
template <> struct GetTypeReverseImpl<glm::i64vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT64_NORM; };
template <> struct GetTypeReverseImpl<glm::u8mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT8_NORM; };
template <> struct GetTypeReverseImpl<glm::i8mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT8_NORM; };
template <> struct GetTypeReverseImpl<glm::u16mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT16_NORM; };
template <> struct GetTypeReverseImpl<glm::i16mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT16_NORM; };
template <> struct GetTypeReverseImpl<glm::u32mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT32_NORM; };
template <> struct GetTypeReverseImpl<glm::i32mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT32_NORM; };
template <> struct GetTypeReverseImpl<glm::u64mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT64_NORM; };
template <> struct GetTypeReverseImpl<glm::i64mat2, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT64_NORM; };
template <> struct GetTypeReverseImpl<glm::u8mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT8_NORM; };
template <> struct GetTypeReverseImpl<glm::i8mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT8_NORM; };
template <> struct GetTypeReverseImpl<glm::u16mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT16_NORM; };
template <> struct GetTypeReverseImpl<glm::i16mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT16_NORM; };
template <> struct GetTypeReverseImpl<glm::u32mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT32_NORM; };
template <> struct GetTypeReverseImpl<glm::i32mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT32_NORM; };
template <> struct GetTypeReverseImpl<glm::u64mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT64_NORM; };
template <> struct GetTypeReverseImpl<glm::i64mat3, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT64_NORM; };
template <> struct GetTypeReverseImpl<glm::u8mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT8_NORM; };
template <> struct GetTypeReverseImpl<glm::i8mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT8_NORM; };
template <> struct GetTypeReverseImpl<glm::u16mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT16_NORM; };
template <> struct GetTypeReverseImpl<glm::i16mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT16_NORM; };
template <> struct GetTypeReverseImpl<glm::u32mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT32_NORM; };
template <> struct GetTypeReverseImpl<glm::i32mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT32_NORM; };
template <> struct GetTypeReverseImpl<glm::u64mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT64_NORM; };
template <> struct GetTypeReverseImpl<glm::i64mat4, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT64_NORM; };
// clang-format on

// Integer primvar lookup in MDL doesn't seem to work so cast all data types to float. This is safe to do since
// FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
// overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways. There is some overhead for UINT8 values
// which could be stored as eUChar.
template <DataType T> struct GetPrimvarTypeImpl;
template <> struct GetPrimvarTypeImpl<DataType::UINT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT32> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT32> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT64> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT64> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::FLOAT64> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT32_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT32_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT64_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT64_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT32> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT32> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT64> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT64> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_FLOAT64> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT32_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT64_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT32> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT32> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT64> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT64> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_FLOAT64> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT32_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT64_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT32> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT32> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT64> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT64> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_FLOAT64> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT32_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT64_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT8> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT8> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT16> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT16> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT32> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT32> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT64> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT64> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::f32mat2; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT8> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT8> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT16> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT16> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT32> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT32> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT64> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT64> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::f32mat3; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT8> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT8> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT16> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT16> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT32> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT32> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT64> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT64> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::f32mat4; };
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::f32mat4; };
template <DataType T> using GetPrimvarType = typename GetPrimvarTypeImpl<T>::Type;

// clang-format off
template <DataType T> struct GetPrimvarBaseDataTypeImpl;
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
// clang-format on

template <typename L, std::size_t... I> const auto& dispatchImpl(std::index_sequence<I...>, L lambda) {
    static decltype(lambda(std::integral_constant<DataType, DataType(0)>{})) array[] = {
        lambda(std::integral_constant<DataType, DataType(I)>{})...};
    return array;
}
template <typename L, typename... P> auto dispatch(L lambda, DataType n, P&&... p) {
    const auto& array = dispatchImpl(std::make_index_sequence<DataTypeCount>{}, lambda);
    return array[static_cast<size_t>(n)](std::forward<P>(p)...);
}

// This allows us to call an enum templated function based on a runtime enum value
#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE(FUNCTION_NAME, TYPE, ...) \
    dispatch([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE, __VA_ARGS__)

// In C++ 20 we don't need this second define
#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(FUNCTION_NAME, TYPE) \
    dispatch([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE)

// TODO: avoid duplicating
template <typename L, std::size_t... I> const auto& dispatchMdlImpl(std::index_sequence<I...>, L lambda) {
    static decltype(lambda(std::integral_constant<MdlInternalPropertyType, MdlInternalPropertyType(0)>{})) array[] = {
        lambda(std::integral_constant<MdlInternalPropertyType, MdlInternalPropertyType(I)>{})...};
    return array;
}
template <typename L, typename... P> auto dispatchMdl(L lambda, MdlInternalPropertyType n, P&&... p) {
    const auto& array = dispatchMdlImpl(std::make_index_sequence<MdlInternalPropertyTypeCount>{}, lambda);
    return array[static_cast<size_t>(n)](std::forward<P>(p)...);
}

#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE(FUNCTION_NAME, TYPE, ...) \
    dispatchMdl([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE, __VA_ARGS__)

#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE_NO_ARGS(FUNCTION_NAME, TYPE) \
    dispatchMdl([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE)

template <DataType T> constexpr bool isNormalized() {
    return IsNormalizedImpl<T>::value;
};

inline bool isNormalized(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(isNormalized, type);
}

template <DataType T> constexpr bool isFloatingPoint() {
    return IsFloatingPointImpl<T>::value;
};

inline bool isFloatingPoint(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(isFloatingPoint, type);
}

template <DataType T> constexpr bool isMatrix() {
    return IsMatrixImpl<T>::value;
};

inline bool isMatrix(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(isMatrix, type);
}

template <DataType T> constexpr MdlInternalPropertyType getMdlInternalPropertyType() {
    return GetMdlInternalPropertyTypeImpl<T>::Type;
};

inline MdlInternalPropertyType getMdlInternalPropertyType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getMdlInternalPropertyType, type);
}

template <DataType T> constexpr uint64_t getComponentCount() {
    return GetComponentCountImpl<T>::ComponentCount;
};

inline uint64_t getComponentCount(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getComponentCount, type);
}

template <typename RawType, typename TransformedType> constexpr DataType getTypeReverse() {
    return GetTypeReverseImpl<RawType, TransformedType>::Type;
};

template <DataType T> constexpr omni::fabric::BaseDataType getPrimvarBaseDataType() {
    return GetPrimvarBaseDataTypeImpl<T>::BaseDataType;
};

inline omni::fabric::BaseDataType getPrimvarBaseDataType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getPrimvarBaseDataType, type);
}

} // namespace cesium::omniverse
