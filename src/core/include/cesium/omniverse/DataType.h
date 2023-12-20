#pragma once

#include <CesiumGltf/PropertyTypeTraits.h>
#include <carb/RenderingTypes.h>
#include <glm/glm.hpp>
#include <omni/fabric/Type.h>

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

// clang-format off
template <DataType T> struct GetNativeTypeImpl;
template <> struct GetNativeTypeImpl<DataType::UINT8> { using Type = uint8_t; };
template <> struct GetNativeTypeImpl<DataType::INT8> { using Type = int8_t; };
template <> struct GetNativeTypeImpl<DataType::UINT16> { using Type = uint16_t; };
template <> struct GetNativeTypeImpl<DataType::INT16> { using Type = int16_t; };
template <> struct GetNativeTypeImpl<DataType::UINT32> { using Type = uint32_t; };
template <> struct GetNativeTypeImpl<DataType::INT32> { using Type = int32_t; };
template <> struct GetNativeTypeImpl<DataType::UINT64> { using Type = uint64_t; };
template <> struct GetNativeTypeImpl<DataType::INT64> { using Type = int64_t; };
template <> struct GetNativeTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetNativeTypeImpl<DataType::FLOAT64> { using Type = double; };
template <> struct GetNativeTypeImpl<DataType::UINT8_NORM> { using Type = uint8_t; };
template <> struct GetNativeTypeImpl<DataType::INT8_NORM> { using Type = int8_t; };
template <> struct GetNativeTypeImpl<DataType::UINT16_NORM> { using Type = uint16_t; };
template <> struct GetNativeTypeImpl<DataType::INT16_NORM> { using Type = int16_t; };
template <> struct GetNativeTypeImpl<DataType::UINT32_NORM> { using Type = uint32_t; };
template <> struct GetNativeTypeImpl<DataType::INT32_NORM> { using Type = int32_t; };
template <> struct GetNativeTypeImpl<DataType::UINT64_NORM> { using Type = uint64_t; };
template <> struct GetNativeTypeImpl<DataType::INT64_NORM> { using Type = int64_t; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT8> { using Type = glm::u8vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT8> { using Type = glm::i8vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT16> { using Type = glm::u16vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT16> { using Type = glm::i16vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT32> { using Type = glm::u32vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT64> { using Type = glm::u64vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT64> { using Type = glm::i64vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_FLOAT64> { using Type = glm::f64vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::u8vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::i8vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::u16vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::i16vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = glm::u32vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT32_NORM> { using Type = glm::i32vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = glm::u64vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC2_INT64_NORM> { using Type = glm::i64vec2; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT8> { using Type = glm::u8vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT8> { using Type = glm::i8vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT16> { using Type = glm::u16vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT16> { using Type = glm::i16vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT32> { using Type = glm::u32vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT64> { using Type = glm::u64vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT64> { using Type = glm::i64vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_FLOAT64> { using Type = glm::f64vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::u8vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::i8vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::u16vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::i16vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = glm::u32vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT32_NORM> { using Type = glm::i32vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = glm::u64vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC3_INT64_NORM> { using Type = glm::i64vec3; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT8> { using Type = glm::u8vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT8> { using Type = glm::i8vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT16> { using Type = glm::u16vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT16> { using Type = glm::i16vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT32> { using Type = glm::u32vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT64> { using Type = glm::u64vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT64> { using Type = glm::i64vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_FLOAT64> { using Type = glm::f64vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::u8vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::i8vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::u16vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::i16vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = glm::u32vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT32_NORM> { using Type = glm::i32vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = glm::u64vec4; };
template <> struct GetNativeTypeImpl<DataType::VEC4_INT64_NORM> { using Type = glm::i64vec4; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT8> { using Type = glm::u8mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT8> { using Type = glm::i8mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT16> { using Type = glm::u16mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT16> { using Type = glm::i16mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT32> { using Type = glm::u32mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT32> { using Type = glm::i32mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT64> { using Type = glm::u64mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT64> { using Type = glm::i64mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f64mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::u8mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::i8mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::u16mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::i16mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::u32mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::i32mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::u64mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::i64mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT8> { using Type = glm::u8mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT8> { using Type = glm::i8mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT16> { using Type = glm::u16mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT16> { using Type = glm::i16mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT32> { using Type = glm::u32mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT32> { using Type = glm::i32mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT64> { using Type = glm::u64mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT64> { using Type = glm::i64mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f64mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::u8mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::i8mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::u16mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::i16mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::u32mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::i32mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::u64mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::i64mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT8> { using Type = glm::u8mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT8> { using Type = glm::i8mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT16> { using Type = glm::u16mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT16> { using Type = glm::i16mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT32> { using Type = glm::u32mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT32> { using Type = glm::i32mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT64> { using Type = glm::u64mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT64> { using Type = glm::i64mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f64mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::u8mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::i8mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::u16mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::i16mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::u32mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::i32mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::u64mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::i64mat4; };
template <DataType T> using GetNativeType = typename GetNativeTypeImpl<T>::Type;

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
template <> struct GetTypeReverseImpl<glm::f64mat2, glm::f64mat2> { static constexpr auto Type = DataType::MAT2_FLOAT64; };
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

template <DataType T> struct GetComponentTypeImpl;
template <> struct GetComponentTypeImpl<DataType::UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetComponentTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetComponentTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = DataType::UINT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto Type = DataType::INT8_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = DataType::UINT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto Type = DataType::INT16_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = DataType::UINT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = DataType::UINT64_NORM;};
template <> struct GetComponentTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto Type = DataType::INT64_NORM;};

template <DataType T> struct GetUnnormalizedTypeImpl;
template <> struct GetUnnormalizedTypeImpl<DataType::UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT8_NORM> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT8_NORM> { static constexpr auto Type = DataType::INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT16_NORM> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT16_NORM> { static constexpr auto Type = DataType::INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT32_NORM> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT32_NORM> { static constexpr auto Type = DataType::INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::UINT64_NORM> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::INT64_NORM> { static constexpr auto Type = DataType::INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT8> { static constexpr auto Type = DataType::VEC2_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT8> { static constexpr auto Type = DataType::VEC2_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT16> { static constexpr auto Type = DataType::VEC2_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT16> { static constexpr auto Type = DataType::VEC2_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT32> { static constexpr auto Type = DataType::VEC2_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT32> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT64> { static constexpr auto Type = DataType::VEC2_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT64> { static constexpr auto Type = DataType::VEC2_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = DataType::VEC2_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto Type = DataType::VEC2_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = DataType::VEC2_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto Type = DataType::VEC2_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = DataType::VEC2_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = DataType::VEC2_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto Type = DataType::VEC2_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT8> { static constexpr auto Type = DataType::VEC3_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT8> { static constexpr auto Type = DataType::VEC3_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT16> { static constexpr auto Type = DataType::VEC3_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT16> { static constexpr auto Type = DataType::VEC3_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT32> { static constexpr auto Type = DataType::VEC3_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT32> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT64> { static constexpr auto Type = DataType::VEC3_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT64> { static constexpr auto Type = DataType::VEC3_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = DataType::VEC3_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto Type = DataType::VEC3_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = DataType::VEC3_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto Type = DataType::VEC3_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = DataType::VEC3_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = DataType::VEC3_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto Type = DataType::VEC3_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT8> { static constexpr auto Type = DataType::VEC4_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT8> { static constexpr auto Type = DataType::VEC4_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT16> { static constexpr auto Type = DataType::VEC4_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT16> { static constexpr auto Type = DataType::VEC4_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT32> { static constexpr auto Type = DataType::VEC4_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT32> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT64> { static constexpr auto Type = DataType::VEC4_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT64> { static constexpr auto Type = DataType::VEC4_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = DataType::VEC4_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto Type = DataType::VEC4_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = DataType::VEC4_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto Type = DataType::VEC4_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = DataType::VEC4_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = DataType::VEC4_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto Type = DataType::VEC4_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT8> { static constexpr auto Type = DataType::MAT2_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT8> { static constexpr auto Type = DataType::MAT2_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT16> { static constexpr auto Type = DataType::MAT2_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT16> { static constexpr auto Type = DataType::MAT2_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT32> { static constexpr auto Type = DataType::MAT2_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT32> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT64> { static constexpr auto Type = DataType::MAT2_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT64> { static constexpr auto Type = DataType::MAT2_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = DataType::MAT2_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto Type = DataType::MAT2_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = DataType::MAT2_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto Type = DataType::MAT2_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = DataType::MAT2_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = DataType::MAT2_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto Type = DataType::MAT2_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT8> { static constexpr auto Type = DataType::MAT3_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT8> { static constexpr auto Type = DataType::MAT3_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT16> { static constexpr auto Type = DataType::MAT3_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT16> { static constexpr auto Type = DataType::MAT3_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT32> { static constexpr auto Type = DataType::MAT3_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT32> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT64> { static constexpr auto Type = DataType::MAT3_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT64> { static constexpr auto Type = DataType::MAT3_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = DataType::MAT3_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto Type = DataType::MAT3_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = DataType::MAT3_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto Type = DataType::MAT3_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = DataType::MAT3_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = DataType::MAT3_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto Type = DataType::MAT3_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT8> { static constexpr auto Type = DataType::MAT4_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT8> { static constexpr auto Type = DataType::MAT4_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT16> { static constexpr auto Type = DataType::MAT4_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT16> { static constexpr auto Type = DataType::MAT4_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT32> { static constexpr auto Type = DataType::MAT4_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT32> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT64> { static constexpr auto Type = DataType::MAT4_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT64> { static constexpr auto Type = DataType::MAT4_INT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = DataType::MAT4_UINT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto Type = DataType::MAT4_INT8;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = DataType::MAT4_UINT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto Type = DataType::MAT4_INT16;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = DataType::MAT4_UINT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = DataType::MAT4_UINT64;};
template <> struct GetUnnormalizedTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto Type = DataType::MAT4_INT64;};

template <DataType T> struct GetTransformedTypeImpl;
template <> struct GetTransformedTypeImpl<DataType::UINT8> { static constexpr auto Type = DataType::UINT8;};
template <> struct GetTransformedTypeImpl<DataType::INT8> { static constexpr auto Type = DataType::INT8;};
template <> struct GetTransformedTypeImpl<DataType::UINT16> { static constexpr auto Type = DataType::UINT16;};
template <> struct GetTransformedTypeImpl<DataType::INT16> { static constexpr auto Type = DataType::INT16;};
template <> struct GetTransformedTypeImpl<DataType::UINT32> { static constexpr auto Type = DataType::UINT32;};
template <> struct GetTransformedTypeImpl<DataType::INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetTransformedTypeImpl<DataType::UINT64> { static constexpr auto Type = DataType::UINT64;};
template <> struct GetTransformedTypeImpl<DataType::INT64> { static constexpr auto Type = DataType::INT64;};
template <> struct GetTransformedTypeImpl<DataType::FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::FLOAT64> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::UINT8_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::INT8_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::UINT16_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::INT16_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::UINT32_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::INT32_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::UINT64_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::INT64_NORM> { static constexpr auto Type = DataType::FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT8> { static constexpr auto Type = DataType::VEC2_UINT8;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT8> { static constexpr auto Type = DataType::VEC2_INT8;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT16> { static constexpr auto Type = DataType::VEC2_UINT16;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT16> { static constexpr auto Type = DataType::VEC2_INT16;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT32> { static constexpr auto Type = DataType::VEC2_UINT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT32> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT64> { static constexpr auto Type = DataType::VEC2_UINT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT64> { static constexpr auto Type = DataType::VEC2_INT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT8> { static constexpr auto Type = DataType::VEC3_UINT8;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT8> { static constexpr auto Type = DataType::VEC3_INT8;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT16> { static constexpr auto Type = DataType::VEC3_UINT16;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT16> { static constexpr auto Type = DataType::VEC3_INT16;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT32> { static constexpr auto Type = DataType::VEC3_UINT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT32> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT64> { static constexpr auto Type = DataType::VEC3_UINT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT64> { static constexpr auto Type = DataType::VEC3_INT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT8> { static constexpr auto Type = DataType::VEC4_UINT8;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT8> { static constexpr auto Type = DataType::VEC4_INT8;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT16> { static constexpr auto Type = DataType::VEC4_UINT16;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT16> { static constexpr auto Type = DataType::VEC4_INT16;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT32> { static constexpr auto Type = DataType::VEC4_UINT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT32> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT64> { static constexpr auto Type = DataType::VEC4_UINT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT64> { static constexpr auto Type = DataType::VEC4_INT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT8> { static constexpr auto Type = DataType::MAT2_UINT8;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT8> { static constexpr auto Type = DataType::MAT2_INT8;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT16> { static constexpr auto Type = DataType::MAT2_UINT16;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT16> { static constexpr auto Type = DataType::MAT2_INT16;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT32> { static constexpr auto Type = DataType::MAT2_UINT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT32> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT64> { static constexpr auto Type = DataType::MAT2_UINT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT64> { static constexpr auto Type = DataType::MAT2_INT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT8> { static constexpr auto Type = DataType::MAT3_UINT8;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT8> { static constexpr auto Type = DataType::MAT3_INT8;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT16> { static constexpr auto Type = DataType::MAT3_UINT16;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT16> { static constexpr auto Type = DataType::MAT3_INT16;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT32> { static constexpr auto Type = DataType::MAT3_UINT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT32> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT64> { static constexpr auto Type = DataType::MAT3_UINT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT64> { static constexpr auto Type = DataType::MAT3_INT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT8> { static constexpr auto Type = DataType::MAT4_UINT8;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT8> { static constexpr auto Type = DataType::MAT4_INT8;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT16> { static constexpr auto Type = DataType::MAT4_UINT16;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT16> { static constexpr auto Type = DataType::MAT4_INT16;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT32> { static constexpr auto Type = DataType::MAT4_UINT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT32> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT64> { static constexpr auto Type = DataType::MAT4_UINT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT64> { static constexpr auto Type = DataType::MAT4_INT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT64;};

// Integer primvar lookup in MDL doesn't seem to work so cast all data types to float. This is safe to do since
// FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
// overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways. There is some overhead for UINT8 values
// which could be stored as eUChar.
template <DataType T> struct GetPrimvarTypeImpl;
template <> struct GetPrimvarTypeImpl<DataType::UINT8> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT8> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT16> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT16> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT64> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT64> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::FLOAT64> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT8_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT8_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT16_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT16_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT32_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT32_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::UINT64_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::INT64_NORM> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT32> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT32> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT64> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT64> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT32> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT32> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT64> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT64> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT32> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT32> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT64> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT64> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT8> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT8> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT16> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT16> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT32> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT32> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT64> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT64> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT8> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT8> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT16> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT16> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT32> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT32> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT64> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT64> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT8> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT8> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT16> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT16> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT32> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT32> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT64> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT64> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetPrimvarTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto Type = DataType::MAT4_FLOAT32;};

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

// eRGB8_UINT, eRGB8_SINT, eRGB16_UINT, eRGB16_SINT do not exist, so use the RGBA equivalent
template <DataType T> struct GetTextureFormatImpl;
template <> struct GetTextureFormatImpl<DataType::UINT8> { static constexpr auto TextureFormat = carb::Format::eR8_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT8> { static constexpr auto TextureFormat = carb::Format::eR8_SINT; };
template <> struct GetTextureFormatImpl<DataType::UINT16> { static constexpr auto TextureFormat = carb::Format::eR16_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT16> { static constexpr auto TextureFormat = carb::Format::eR16_SINT; };
template <> struct GetTextureFormatImpl<DataType::UINT32> { static constexpr auto TextureFormat = carb::Format::eR32_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT32> { static constexpr auto TextureFormat = carb::Format::eR32_SINT; };
template <> struct GetTextureFormatImpl<DataType::UINT64> { static constexpr auto TextureFormat = carb::Format::eR32_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT64> { static constexpr auto TextureFormat = carb::Format::eR32_SINT; };
template <> struct GetTextureFormatImpl<DataType::FLOAT32> { static constexpr auto TextureFormat = carb::Format::eR32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::FLOAT64> { static constexpr auto TextureFormat = carb::Format::eR32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eR8_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eR8_SINT; };
template <> struct GetTextureFormatImpl<DataType::UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eR16_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eR16_SINT; };
template <> struct GetTextureFormatImpl<DataType::UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eR32_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eR32_SINT; };
template <> struct GetTextureFormatImpl<DataType::UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eR32_UINT; };
template <> struct GetTextureFormatImpl<DataType::INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eR32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT8> { static constexpr auto TextureFormat = carb::Format::eRG8_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT8> { static constexpr auto TextureFormat = carb::Format::eRG8_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT16> { static constexpr auto TextureFormat = carb::Format::eRG16_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT16> { static constexpr auto TextureFormat = carb::Format::eRG16_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT32> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT32> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT64> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT64> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_FLOAT32> { static constexpr auto TextureFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_FLOAT64> { static constexpr auto TextureFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRG8_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRG8_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRG16_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRG16_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC2_INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT32> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT32> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT64> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT64> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_FLOAT32> { static constexpr auto TextureFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_FLOAT64> { static constexpr auto TextureFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC3_INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT32> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT32> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT64> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT64> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_FLOAT32> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_FLOAT64> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::VEC4_INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT8> { static constexpr auto TextureFormat = carb::Format::eRG8_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT8> { static constexpr auto TextureFormat = carb::Format::eRG8_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT16> { static constexpr auto TextureFormat = carb::Format::eRG16_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT16> { static constexpr auto TextureFormat = carb::Format::eRG16_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT32> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT32> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT64> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT64> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_FLOAT32> { static constexpr auto TextureFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_FLOAT64> { static constexpr auto TextureFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRG8_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRG8_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRG16_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRG16_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT2_INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRG32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT32> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT32> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT64> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT64> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_FLOAT32> { static constexpr auto TextureFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_FLOAT64> { static constexpr auto TextureFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT3_INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGB32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT8> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT16> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT32> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT32> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT64> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT64> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_FLOAT32> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_FLOAT64> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT8_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT16_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT32_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetTextureFormatImpl<DataType::MAT4_INT64_NORM> { static constexpr auto TextureFormat = carb::Format::eRGBA32_SINT; };

// MDL doesn't support reading from integer textures, so integer values must be converted to float. See note in cesium.mdl for more details.
template <DataType T> struct GetPropertyTableTextureTypeImpl;
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT8> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT8> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT16> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT16> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT32> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT32> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT64> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT64> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::FLOAT32> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::FLOAT64> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT8_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT8_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT16_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT16_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT32_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT32_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::UINT64_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::INT64_NORM> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT8> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT8> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT16> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT16> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT32> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT32> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT64> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT64> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT8> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT8> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT16> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT16> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT32> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT32> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT64> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT64> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT8> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT8> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT16> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT16> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT32> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT32> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT64> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT64> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT8> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT8> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT16> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT16> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT32> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT32> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT64> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT64> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT8> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT8> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT16> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT16> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT32> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT32> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT64> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT64> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT8> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT8> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT16> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT16> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT32> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT32> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT64> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT64> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetPropertyTableTextureTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto Type = DataType::VEC4_FLOAT32; };

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

template <typename T, typename L, std::size_t... I> const auto& dispatchImpl(std::index_sequence<I...>, L lambda) {
    static decltype(lambda(std::integral_constant<T, T(0)>{})) array[] = {lambda(std::integral_constant<T, T(I)>{})...};
    return array;
}
template <uint64_t T_COUNT, typename T, typename L, typename... P> auto dispatch(L lambda, T n, P&&... p) {
    const auto& array = dispatchImpl<T>(std::make_index_sequence<T_COUNT>{}, lambda);
    return array[static_cast<size_t>(n)](std::forward<P>(p)...);
}

// This allows us to call an enum templated function based on a runtime enum value
#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE(FUNCTION_NAME, TYPE, ...) \
    dispatch<DataTypeCount>([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE, __VA_ARGS__)

// In C++ 20 we don't need this second define
#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(FUNCTION_NAME, TYPE) \
    dispatch<DataTypeCount>([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE)

#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE(FUNCTION_NAME, TYPE, ...) \
    dispatch<MdlInternalPropertyTypeCount>([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE, __VA_ARGS__)

#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_MDL_TYPE_NO_ARGS(FUNCTION_NAME, TYPE) \
    dispatch<MdlInternalPropertyTypeCount>([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE)

template <typename RawType, typename TransformedType> constexpr DataType getTypeReverse() {
    return GetTypeReverseImpl<RawType, TransformedType>::Type;
};

template <DataType T> constexpr DataType getComponentType() {
    return GetComponentTypeImpl<T>::Type;
};

inline DataType getComponentType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getComponentType, type);
}

template <DataType T> constexpr DataType getUnnormalizedType() {
    return GetUnnormalizedTypeImpl<T>::Type;
};

inline DataType getUnnormalizedType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getUnnormalizedType, type);
}

template <DataType T> constexpr DataType getUnnormalizedComponentType() {
    return getUnnormalizedType<getComponentType<T>()>();
};

inline DataType getUnnormalizedComponentType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getUnnormalizedComponentType, type);
}

template <DataType T> constexpr DataType getTransformedType() {
    return GetTransformedTypeImpl<T>::Type;
};

inline DataType getTransformedType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getTransformedType, type);
};

template <DataType T> constexpr DataType getPrimvarType() {
    return GetPrimvarTypeImpl<T>::Type;
};

inline DataType getPrimvarType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getPrimvarType, type);
};

template <DataType T> constexpr omni::fabric::BaseDataType getPrimvarBaseDataType() {
    return GetPrimvarBaseDataTypeImpl<T>::BaseDataType;
};

inline omni::fabric::BaseDataType getPrimvarBaseDataType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getPrimvarBaseDataType, type);
}

template <DataType T> constexpr carb::Format getTextureFormat() {
    return GetTextureFormatImpl<T>::TextureFormat;
};

inline carb::Format getTextureFormat(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getTextureFormat, type);
};

template <DataType T> constexpr DataType getPropertyTableTextureType() {
    return GetPropertyTableTextureTypeImpl<T>::Type;
};

inline DataType getPropertyTableTextureType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getPropertyTableTextureType, type);
};

template <DataType T> constexpr MdlInternalPropertyType getMdlInternalPropertyType() {
    return GetMdlInternalPropertyTypeImpl<T>::Type;
};

inline MdlInternalPropertyType getMdlInternalPropertyType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getMdlInternalPropertyType, type);
}

template <DataType T> constexpr bool isVector() {
    return CesiumGltf::IsMetadataVecN<GetNativeType<T>>();
};

inline bool isVector(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(isVector, type);
}

template <DataType T> constexpr bool isMatrix() {
    return CesiumGltf::IsMetadataMatN<GetNativeType<T>>();
};

inline bool isMatrix(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(isMatrix, type);
}

template <DataType T> constexpr uint64_t getComponentCount() {
    if constexpr (isMatrix<T>() || isVector<T>()) {
        return GetNativeType<T>::length();
    } else {
        return 1;
    }
};

inline uint64_t getComponentCount(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getComponentCount, type);
}

template <DataType T> constexpr bool isNormalized() {
    return getUnnormalizedType<T>() != T;
};

inline bool isNormalized(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(isNormalized, type);
}

template <DataType T> constexpr uint64_t getByteLength() {
    return sizeof(GetNativeType<T>);
};

inline uint64_t getByteLength(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getByteLength, type);
};

template <DataType T> constexpr uint64_t getComponentByteLength() {
    using NativeComponentType = GetNativeType<getComponentType<T>()>;
    return sizeof(NativeComponentType);
};

inline uint64_t getComponentByteLength(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getComponentByteLength, type);
};

} // namespace cesium::omniverse
