#pragma once

#include "cesium/omniverse/DataType.h"

#include <CesiumGltf/Accessor.h>
#include <CesiumGltf/ClassProperty.h>
#include <CesiumGltf/PropertyType.h>
#include <CesiumGltf/Schema.h>
#include <glm/glm.hpp>
#include <omni/fabric/FabricTypes.h>

#include <optional>
#include <string>

namespace cesium::omniverse {

enum class TypeGroup {
    SCALAR,
    VECTOR,
    MATRIX,
    UNKNOWN,
};

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
    UNKNOWN,
};

template <DataType T> struct IsNormalized;
template <> struct IsNormalized<DataType::UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::INT8> : std::false_type {};
template <> struct IsNormalized<DataType::UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::INT16> : std::false_type {};
template <> struct IsNormalized<DataType::UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::INT32> : std::false_type {};
template <> struct IsNormalized<DataType::UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::INT64> : std::false_type {};
template <> struct IsNormalized<DataType::FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::INT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_INT8> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_INT16> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_INT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_INT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC2_UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC2_INT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_INT8> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_INT16> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_INT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_INT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC3_UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC3_INT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_INT8> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_INT16> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_INT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_INT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::VEC4_UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::VEC4_INT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_INT8> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_INT16> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_INT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_INT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT2_UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT2_INT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_INT8> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_INT16> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_INT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_INT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT3_UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT3_INT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_UINT8> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_INT8> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_UINT16> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_INT16> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_UINT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_INT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_UINT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_INT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_FLOAT32> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_FLOAT64> : std::false_type {};
template <> struct IsNormalized<DataType::MAT4_UINT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_INT8_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_UINT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_INT16_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_UINT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_INT32_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_UINT64_NORM> : std::true_type {};
template <> struct IsNormalized<DataType::MAT4_INT64_NORM> : std::true_type {};

// GLM does not typedef all integer matrix types, so they are written in full.
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
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT8> { using Type = glm::mat<2, 2, uint8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT8> { using Type = glm::mat<2, 2, int8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT16> { using Type = glm::mat<2, 2, uint16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT16> { using Type = glm::mat<2, 2, int16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT32> { using Type = glm::mat<2, 2, uint32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT32> { using Type = glm::mat<2, 2, int32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT64> { using Type = glm::mat<2, 2, uint64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT64> { using Type = glm::mat<2, 2, int64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f64mat2; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::mat<2, 2, uint8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::mat<2, 2, int8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::mat<2, 2, uint16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::mat<2, 2, int16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::mat<2, 2, uint32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::mat<2, 2, int32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::mat<2, 2, uint64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::mat<2, 2, int64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT8> { using Type = glm::mat<3, 3, uint8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT8> { using Type = glm::mat<3, 3, int8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT16> { using Type = glm::mat<3, 3, uint16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT16> { using Type = glm::mat<3, 3, int16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT32> { using Type = glm::mat<3, 3, uint32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT32> { using Type = glm::mat<3, 3, int32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT64> { using Type = glm::mat<3, 3, uint64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT64> { using Type = glm::mat<3, 3, int64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f64mat3; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::mat<3, 3, uint8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::mat<3, 3, int8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::mat<3, 3, uint16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::mat<3, 3, int16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::mat<3, 3, uint32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::mat<3, 3, int32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::mat<3, 3, uint64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::mat<3, 3, int64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT8> { using Type = glm::mat<4, 4, uint8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT8> { using Type = glm::mat<4, 4, int8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT16> { using Type = glm::mat<4, 4, uint16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT16> { using Type = glm::mat<4, 4, int16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT32> { using Type = glm::mat<4, 4, uint32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT32> { using Type = glm::mat<4, 4, int32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT64> { using Type = glm::mat<4, 4, uint64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT64> { using Type = glm::mat<4, 4, int64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f64mat4; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::mat<4, 4, uint8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::mat<4, 4, int8_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::mat<4, 4, uint16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::mat<4, 4, int16_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::mat<4, 4, uint32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::mat<4, 4, int32_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::mat<4, 4, uint64_t>; };
template <> struct GetNativeTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::mat<4, 4, int64_t>; };

template <DataType T> using GetNativeType = typename GetNativeTypeImpl<T>::Type;

// clang-format off
template <typename T> struct GetNativeTypeReverse;
template <> struct GetNativeTypeReverse<uint8_t> { static constexpr auto Type = DataType::UINT8; };
template <> struct GetNativeTypeReverse<int8_t> { static constexpr auto Type = DataType::INT8; };
template <> struct GetNativeTypeReverse<uint16_t> { static constexpr auto Type = DataType::UINT16; };
template <> struct GetNativeTypeReverse<int16_t> { static constexpr auto Type = DataType::INT16; };
template <> struct GetNativeTypeReverse<uint32_t> { static constexpr auto Type = DataType::UINT32; };
template <> struct GetNativeTypeReverse<int32_t> { static constexpr auto Type = DataType::INT32; };
template <> struct GetNativeTypeReverse<uint64_t> { static constexpr auto Type = DataType::UINT64; };
template <> struct GetNativeTypeReverse<int64_t> { static constexpr auto Type = DataType::INT64; };
template <> struct GetNativeTypeReverse<float> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetNativeTypeReverse<double> { static constexpr auto Type = DataType::FLOAT64; };
template <> struct GetNativeTypeReverse<glm::u8vec2> { static constexpr auto Type = DataType::VEC2_UINT8; };
template <> struct GetNativeTypeReverse<glm::i8vec2> { static constexpr auto Type = DataType::VEC2_INT8; };
template <> struct GetNativeTypeReverse<glm::u16vec2> { static constexpr auto Type = DataType::VEC2_UINT16; };
template <> struct GetNativeTypeReverse<glm::i16vec2> { static constexpr auto Type = DataType::VEC2_INT16; };
template <> struct GetNativeTypeReverse<glm::u32vec2> { static constexpr auto Type = DataType::VEC2_UINT32; };
template <> struct GetNativeTypeReverse<glm::i32vec2> { static constexpr auto Type = DataType::VEC2_INT32; };
template <> struct GetNativeTypeReverse<glm::u64vec2> { static constexpr auto Type = DataType::VEC2_UINT64; };
template <> struct GetNativeTypeReverse<glm::i64vec2> { static constexpr auto Type = DataType::VEC2_INT64; };
template <> struct GetNativeTypeReverse<glm::f32vec2> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetNativeTypeReverse<glm::f64vec2> { static constexpr auto Type = DataType::VEC2_FLOAT64; };
template <> struct GetNativeTypeReverse<glm::u8vec3> { static constexpr auto Type = DataType::VEC3_UINT8; };
template <> struct GetNativeTypeReverse<glm::i8vec3> { static constexpr auto Type = DataType::VEC3_INT8; };
template <> struct GetNativeTypeReverse<glm::u16vec3> { static constexpr auto Type = DataType::VEC3_UINT16; };
template <> struct GetNativeTypeReverse<glm::i16vec3> { static constexpr auto Type = DataType::VEC3_INT16; };
template <> struct GetNativeTypeReverse<glm::u32vec3> { static constexpr auto Type = DataType::VEC3_UINT32; };
template <> struct GetNativeTypeReverse<glm::i32vec3> { static constexpr auto Type = DataType::VEC3_INT32; };
template <> struct GetNativeTypeReverse<glm::u64vec3> { static constexpr auto Type = DataType::VEC3_UINT64; };
template <> struct GetNativeTypeReverse<glm::i64vec3> { static constexpr auto Type = DataType::VEC3_INT64; };
template <> struct GetNativeTypeReverse<glm::f32vec3> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetNativeTypeReverse<glm::f64vec3> { static constexpr auto Type = DataType::VEC3_FLOAT64; };
template <> struct GetNativeTypeReverse<glm::u8vec4> { static constexpr auto Type = DataType::VEC4_UINT8; };
template <> struct GetNativeTypeReverse<glm::i8vec4> { static constexpr auto Type = DataType::VEC4_INT8; };
template <> struct GetNativeTypeReverse<glm::u16vec4> { static constexpr auto Type = DataType::VEC4_UINT16; };
template <> struct GetNativeTypeReverse<glm::i16vec4> { static constexpr auto Type = DataType::VEC4_INT16; };
template <> struct GetNativeTypeReverse<glm::u32vec4> { static constexpr auto Type = DataType::VEC4_UINT32; };
template <> struct GetNativeTypeReverse<glm::i32vec4> { static constexpr auto Type = DataType::VEC4_INT32; };
template <> struct GetNativeTypeReverse<glm::u64vec4> { static constexpr auto Type = DataType::VEC4_UINT64; };
template <> struct GetNativeTypeReverse<glm::i64vec4> { static constexpr auto Type = DataType::VEC4_INT64; };
template <> struct GetNativeTypeReverse<glm::f32vec4> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetNativeTypeReverse<glm::f64vec4> { static constexpr auto Type = DataType::VEC4_FLOAT64; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, uint8_t>> { static constexpr auto Type = DataType::MAT2_UINT8; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, int8_t>> { static constexpr auto Type = DataType::MAT2_INT8; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, uint16_t>> { static constexpr auto Type = DataType::MAT2_UINT16; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, int16_t>> { static constexpr auto Type = DataType::MAT2_INT16; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, uint32_t>> { static constexpr auto Type = DataType::MAT2_UINT32; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, int32_t>> { static constexpr auto Type = DataType::MAT2_INT32; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, uint64_t>> { static constexpr auto Type = DataType::MAT2_UINT64; };
template <> struct GetNativeTypeReverse<glm::mat<2, 2, int64_t>> { static constexpr auto Type = DataType::MAT2_INT64; };
template <> struct GetNativeTypeReverse<glm::f32mat2> { static constexpr auto Type = DataType::MAT2_FLOAT32; };
template <> struct GetNativeTypeReverse<glm::f64mat2> { static constexpr auto Type = DataType::MAT2_FLOAT64; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, uint8_t>> { static constexpr auto Type = DataType::MAT3_UINT8; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, int8_t>> { static constexpr auto Type = DataType::MAT3_INT8; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, uint16_t>> { static constexpr auto Type = DataType::MAT3_UINT16; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, int16_t>> { static constexpr auto Type = DataType::MAT3_INT16; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, uint32_t>> { static constexpr auto Type = DataType::MAT3_UINT32; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, int32_t>> { static constexpr auto Type = DataType::MAT3_INT32; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, uint64_t>> { static constexpr auto Type = DataType::MAT3_UINT64; };
template <> struct GetNativeTypeReverse<glm::mat<3, 3, int64_t>> { static constexpr auto Type = DataType::MAT3_INT64; };
template <> struct GetNativeTypeReverse<glm::f32mat3> { static constexpr auto Type = DataType::MAT3_FLOAT32; };
template <> struct GetNativeTypeReverse<glm::f64mat3> { static constexpr auto Type = DataType::MAT3_FLOAT64; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, uint8_t>> { static constexpr auto Type = DataType::MAT4_UINT8; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, int8_t>> { static constexpr auto Type = DataType::MAT4_INT8; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, uint16_t>> { static constexpr auto Type = DataType::MAT4_UINT16; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, int16_t>> { static constexpr auto Type = DataType::MAT4_INT16; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, uint32_t>> { static constexpr auto Type = DataType::MAT4_UINT32; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, int32_t>> { static constexpr auto Type = DataType::MAT4_INT32; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, uint64_t>> { static constexpr auto Type = DataType::MAT4_UINT64; };
template <> struct GetNativeTypeReverse<glm::mat<4, 4, int64_t>> { static constexpr auto Type = DataType::MAT4_INT64; };
template <> struct GetNativeTypeReverse<glm::f32mat4> { static constexpr auto Type = DataType::MAT4_FLOAT32; };
template <> struct GetNativeTypeReverse<glm::f64mat4> { static constexpr auto Type = DataType::MAT4_FLOAT64; };
// clang-format on

// Only specialized for valid glTF vertex attribute types and valid fabric primvar types.
// Excludes 32 and 64-bit integer types, 64-bit floating point types, and matrix types.
template <DataType T> struct GetPrimvarTypeImpl;

// Integer primvar lookup in MDL doesn't seem to work so cast all data types to float. This is safe to do since
// FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
// overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways. There is some overhead for UINT8 values
// which could be stored as the eUChar type.
template <> struct GetPrimvarTypeImpl<DataType::UINT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
// Normalized types are always treated as float types since they are normalized on the CPU prior to being uploaded
// to Fabric. In the future we may do normalization in MDL.
template <> struct GetPrimvarTypeImpl<DataType::UINT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::f32vec4; };

template <DataType T> using GetPrimvarType = typename GetPrimvarTypeImpl<T>::Type;

constexpr DataType getComponentType(DataType type) {
    // clang-format off
    switch (type) {
        case DataType::UINT8: return DataType::UINT8;
        case DataType::INT8: return DataType::INT8;
        case DataType::UINT16: return DataType::UINT16;
        case DataType::INT16: return DataType::INT16;
        case DataType::UINT32: return DataType::UINT32;
        case DataType::INT32: return DataType::INT32;
        case DataType::UINT64: return DataType::UINT64;
        case DataType::INT64: return DataType::INT64;
        case DataType::FLOAT32: return DataType::FLOAT32;
        case DataType::FLOAT64: return DataType::FLOAT64;
        case DataType::UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::INT8_NORM: return DataType::INT8_NORM;
        case DataType::UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::INT16_NORM: return DataType::INT16_NORM;
        case DataType::UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::INT32_NORM: return DataType::INT32_NORM;
        case DataType::UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::INT64_NORM: return DataType::INT64_NORM;
        case DataType::VEC2_UINT8: return DataType::UINT8;
        case DataType::VEC2_INT8: return DataType::INT8;
        case DataType::VEC2_UINT16: return DataType::UINT16;
        case DataType::VEC2_INT16: return DataType::INT16;
        case DataType::VEC2_UINT32: return DataType::UINT32;
        case DataType::VEC2_INT32: return DataType::INT32;
        case DataType::VEC2_UINT64: return DataType::UINT64;
        case DataType::VEC2_INT64: return DataType::INT64;
        case DataType::VEC2_FLOAT32: return DataType::FLOAT32;
        case DataType::VEC2_FLOAT64: return DataType::FLOAT64;
        case DataType::VEC2_UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::VEC2_INT8_NORM: return DataType::INT8_NORM;
        case DataType::VEC2_UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::VEC2_INT16_NORM: return DataType::INT16_NORM;
        case DataType::VEC2_UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::VEC2_INT32_NORM: return DataType::INT32_NORM;
        case DataType::VEC2_UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::VEC2_INT64_NORM: return DataType::INT64_NORM;
        case DataType::VEC3_UINT8: return DataType::UINT8;
        case DataType::VEC3_INT8: return DataType::INT8;
        case DataType::VEC3_UINT16: return DataType::UINT16;
        case DataType::VEC3_INT16: return DataType::INT16;
        case DataType::VEC3_UINT32: return DataType::UINT32;
        case DataType::VEC3_INT32: return DataType::INT32;
        case DataType::VEC3_UINT64: return DataType::UINT64;
        case DataType::VEC3_INT64: return DataType::INT64;
        case DataType::VEC3_FLOAT32: return DataType::FLOAT32;
        case DataType::VEC3_FLOAT64: return DataType::FLOAT64;
        case DataType::VEC3_UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::VEC3_INT8_NORM: return DataType::INT8_NORM;
        case DataType::VEC3_UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::VEC3_INT16_NORM: return DataType::INT16_NORM;
        case DataType::VEC3_UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::VEC3_INT32_NORM: return DataType::INT32_NORM;
        case DataType::VEC3_UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::VEC3_INT64_NORM: return DataType::INT64_NORM;
        case DataType::VEC4_UINT8: return DataType::UINT8;
        case DataType::VEC4_INT8: return DataType::INT8;
        case DataType::VEC4_UINT16: return DataType::UINT16;
        case DataType::VEC4_INT16: return DataType::INT16;
        case DataType::VEC4_UINT32: return DataType::UINT32;
        case DataType::VEC4_INT32: return DataType::INT32;
        case DataType::VEC4_UINT64: return DataType::UINT64;
        case DataType::VEC4_INT64: return DataType::INT64;
        case DataType::VEC4_FLOAT32: return DataType::FLOAT32;
        case DataType::VEC4_FLOAT64: return DataType::FLOAT64;
        case DataType::VEC4_UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::VEC4_INT8_NORM: return DataType::INT8_NORM;
        case DataType::VEC4_UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::VEC4_INT16_NORM: return DataType::INT16_NORM;
        case DataType::VEC4_UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::VEC4_INT32_NORM: return DataType::INT32_NORM;
        case DataType::VEC4_UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::VEC4_INT64_NORM: return DataType::INT64_NORM;
        case DataType::MAT2_UINT8: return DataType::UINT8;
        case DataType::MAT2_INT8: return DataType::INT8;
        case DataType::MAT2_UINT16: return DataType::UINT16;
        case DataType::MAT2_INT16: return DataType::INT16;
        case DataType::MAT2_UINT32: return DataType::UINT32;
        case DataType::MAT2_INT32: return DataType::INT32;
        case DataType::MAT2_UINT64: return DataType::UINT64;
        case DataType::MAT2_INT64: return DataType::INT64;
        case DataType::MAT2_FLOAT32: return DataType::FLOAT32;
        case DataType::MAT2_FLOAT64: return DataType::FLOAT64;
        case DataType::MAT2_UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::MAT2_INT8_NORM: return DataType::INT8_NORM;
        case DataType::MAT2_UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::MAT2_INT16_NORM: return DataType::INT16_NORM;
        case DataType::MAT2_UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::MAT2_INT32_NORM: return DataType::INT32_NORM;
        case DataType::MAT2_UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::MAT2_INT64_NORM: return DataType::INT64_NORM;
        case DataType::MAT3_UINT8: return DataType::UINT8;
        case DataType::MAT3_INT8: return DataType::INT8;
        case DataType::MAT3_UINT16: return DataType::UINT16;
        case DataType::MAT3_INT16: return DataType::INT16;
        case DataType::MAT3_UINT32: return DataType::UINT32;
        case DataType::MAT3_INT32: return DataType::INT32;
        case DataType::MAT3_UINT64: return DataType::UINT64;
        case DataType::MAT3_INT64: return DataType::INT64;
        case DataType::MAT3_FLOAT32: return DataType::FLOAT32;
        case DataType::MAT3_FLOAT64: return DataType::FLOAT64;
        case DataType::MAT3_UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::MAT3_INT8_NORM: return DataType::INT8_NORM;
        case DataType::MAT3_UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::MAT3_INT16_NORM: return DataType::INT16_NORM;
        case DataType::MAT3_UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::MAT3_INT32_NORM: return DataType::INT32_NORM;
        case DataType::MAT3_UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::MAT3_INT64_NORM: return DataType::INT64_NORM;
        case DataType::MAT4_UINT8: return DataType::UINT8;
        case DataType::MAT4_INT8: return DataType::INT8;
        case DataType::MAT4_UINT16: return DataType::UINT16;
        case DataType::MAT4_INT16: return DataType::INT16;
        case DataType::MAT4_UINT32: return DataType::UINT32;
        case DataType::MAT4_INT32: return DataType::INT32;
        case DataType::MAT4_UINT64: return DataType::UINT64;
        case DataType::MAT4_INT64: return DataType::INT64;
        case DataType::MAT4_FLOAT32: return DataType::FLOAT32;
        case DataType::MAT4_FLOAT64: return DataType::FLOAT64;
        case DataType::MAT4_UINT8_NORM: return DataType::UINT8_NORM;
        case DataType::MAT4_INT8_NORM: return DataType::INT8_NORM;
        case DataType::MAT4_UINT16_NORM: return DataType::UINT16_NORM;
        case DataType::MAT4_INT16_NORM: return DataType::INT16_NORM;
        case DataType::MAT4_UINT32_NORM: return DataType::UINT32_NORM;
        case DataType::MAT4_INT32_NORM: return DataType::INT32_NORM;
        case DataType::MAT4_UINT64_NORM: return DataType::UINT64_NORM;
        case DataType::MAT4_INT64_NORM: return DataType::INT64_NORM;
        case DataType::UNKNOWN: return DataType::UNKNOWN;
    }
    // clang-format on

    // Unreachable code. All enum cases are handled above.
    assert(false);
    return DataType::UNKNOWN;
}

constexpr uint64_t getComponentCount(DataType type) {
    switch (type) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FLOAT32:
        case DataType::FLOAT64:
        case DataType::UINT8_NORM:
        case DataType::INT8_NORM:
        case DataType::UINT16_NORM:
        case DataType::INT16_NORM:
        case DataType::UINT32_NORM:
        case DataType::INT32_NORM:
        case DataType::UINT64_NORM:
        case DataType::INT64_NORM:
            return 1;
        case DataType::VEC2_UINT8:
        case DataType::VEC2_INT8:
        case DataType::VEC2_UINT16:
        case DataType::VEC2_INT16:
        case DataType::VEC2_UINT32:
        case DataType::VEC2_INT32:
        case DataType::VEC2_UINT64:
        case DataType::VEC2_INT64:
        case DataType::VEC2_FLOAT32:
        case DataType::VEC2_FLOAT64:
        case DataType::VEC2_UINT8_NORM:
        case DataType::VEC2_INT8_NORM:
        case DataType::VEC2_UINT16_NORM:
        case DataType::VEC2_INT16_NORM:
        case DataType::VEC2_UINT32_NORM:
        case DataType::VEC2_INT32_NORM:
        case DataType::VEC2_UINT64_NORM:
        case DataType::VEC2_INT64_NORM:
            return 2;
        case DataType::VEC3_UINT8:
        case DataType::VEC3_INT8:
        case DataType::VEC3_UINT16:
        case DataType::VEC3_INT16:
        case DataType::VEC3_UINT32:
        case DataType::VEC3_INT32:
        case DataType::VEC3_UINT64:
        case DataType::VEC3_INT64:
        case DataType::VEC3_FLOAT32:
        case DataType::VEC3_FLOAT64:
        case DataType::VEC3_UINT8_NORM:
        case DataType::VEC3_INT8_NORM:
        case DataType::VEC3_UINT16_NORM:
        case DataType::VEC3_INT16_NORM:
        case DataType::VEC3_UINT32_NORM:
        case DataType::VEC3_INT32_NORM:
        case DataType::VEC3_UINT64_NORM:
        case DataType::VEC3_INT64_NORM:
            return 3;
        case DataType::VEC4_UINT8:
        case DataType::VEC4_INT8:
        case DataType::VEC4_UINT16:
        case DataType::VEC4_INT16:
        case DataType::VEC4_UINT32:
        case DataType::VEC4_INT32:
        case DataType::VEC4_UINT64:
        case DataType::VEC4_INT64:
        case DataType::VEC4_FLOAT32:
        case DataType::VEC4_FLOAT64:
        case DataType::VEC4_UINT8_NORM:
        case DataType::VEC4_INT8_NORM:
        case DataType::VEC4_UINT16_NORM:
        case DataType::VEC4_INT16_NORM:
        case DataType::VEC4_UINT32_NORM:
        case DataType::VEC4_INT32_NORM:
        case DataType::VEC4_UINT64_NORM:
        case DataType::VEC4_INT64_NORM:
        case DataType::MAT2_UINT8:
        case DataType::MAT2_INT8:
        case DataType::MAT2_UINT16:
        case DataType::MAT2_INT16:
        case DataType::MAT2_UINT32:
        case DataType::MAT2_INT32:
        case DataType::MAT2_UINT64:
        case DataType::MAT2_INT64:
        case DataType::MAT2_FLOAT32:
        case DataType::MAT2_FLOAT64:
        case DataType::MAT2_UINT8_NORM:
        case DataType::MAT2_INT8_NORM:
        case DataType::MAT2_UINT16_NORM:
        case DataType::MAT2_INT16_NORM:
        case DataType::MAT2_UINT32_NORM:
        case DataType::MAT2_INT32_NORM:
        case DataType::MAT2_UINT64_NORM:
        case DataType::MAT2_INT64_NORM:
            return 4;
        case DataType::MAT3_UINT8:
        case DataType::MAT3_INT8:
        case DataType::MAT3_UINT16:
        case DataType::MAT3_INT16:
        case DataType::MAT3_UINT32:
        case DataType::MAT3_INT32:
        case DataType::MAT3_UINT64:
        case DataType::MAT3_INT64:
        case DataType::MAT3_FLOAT32:
        case DataType::MAT3_FLOAT64:
        case DataType::MAT3_UINT8_NORM:
        case DataType::MAT3_INT8_NORM:
        case DataType::MAT3_UINT16_NORM:
        case DataType::MAT3_INT16_NORM:
        case DataType::MAT3_UINT32_NORM:
        case DataType::MAT3_INT32_NORM:
        case DataType::MAT3_UINT64_NORM:
        case DataType::MAT3_INT64_NORM:
            return 9;
        case DataType::MAT4_UINT8:
        case DataType::MAT4_INT8:
        case DataType::MAT4_UINT16:
        case DataType::MAT4_INT16:
        case DataType::MAT4_UINT32:
        case DataType::MAT4_INT32:
        case DataType::MAT4_UINT64:
        case DataType::MAT4_INT64:
        case DataType::MAT4_FLOAT32:
        case DataType::MAT4_FLOAT64:
        case DataType::MAT4_UINT8_NORM:
        case DataType::MAT4_INT8_NORM:
        case DataType::MAT4_UINT16_NORM:
        case DataType::MAT4_INT16_NORM:
        case DataType::MAT4_UINT32_NORM:
        case DataType::MAT4_INT32_NORM:
        case DataType::MAT4_UINT64_NORM:
        case DataType::MAT4_INT64_NORM:
            return 16;
        case DataType::UNKNOWN:
            return 0;
    }

    // Unreachable code. All enum cases are handled above.
    assert(false);
    return 0;
}

constexpr TypeGroup getGroup(DataType type) {
    switch (type) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FLOAT32:
        case DataType::FLOAT64:
        case DataType::UINT8_NORM:
        case DataType::INT8_NORM:
        case DataType::UINT16_NORM:
        case DataType::INT16_NORM:
        case DataType::UINT32_NORM:
        case DataType::INT32_NORM:
        case DataType::UINT64_NORM:
        case DataType::INT64_NORM:
            return TypeGroup::SCALAR;
        case DataType::VEC2_UINT8:
        case DataType::VEC2_INT8:
        case DataType::VEC2_UINT16:
        case DataType::VEC2_INT16:
        case DataType::VEC2_UINT32:
        case DataType::VEC2_INT32:
        case DataType::VEC2_UINT64:
        case DataType::VEC2_INT64:
        case DataType::VEC2_FLOAT32:
        case DataType::VEC2_FLOAT64:
        case DataType::VEC2_UINT8_NORM:
        case DataType::VEC2_INT8_NORM:
        case DataType::VEC2_UINT16_NORM:
        case DataType::VEC2_INT16_NORM:
        case DataType::VEC2_UINT32_NORM:
        case DataType::VEC2_INT32_NORM:
        case DataType::VEC2_UINT64_NORM:
        case DataType::VEC2_INT64_NORM:
        case DataType::VEC3_UINT8:
        case DataType::VEC3_INT8:
        case DataType::VEC3_UINT16:
        case DataType::VEC3_INT16:
        case DataType::VEC3_UINT32:
        case DataType::VEC3_INT32:
        case DataType::VEC3_UINT64:
        case DataType::VEC3_INT64:
        case DataType::VEC3_FLOAT32:
        case DataType::VEC3_FLOAT64:
        case DataType::VEC3_UINT8_NORM:
        case DataType::VEC3_INT8_NORM:
        case DataType::VEC3_UINT16_NORM:
        case DataType::VEC3_INT16_NORM:
        case DataType::VEC3_UINT32_NORM:
        case DataType::VEC3_INT32_NORM:
        case DataType::VEC3_UINT64_NORM:
        case DataType::VEC3_INT64_NORM:
        case DataType::VEC4_UINT8:
        case DataType::VEC4_INT8:
        case DataType::VEC4_UINT16:
        case DataType::VEC4_INT16:
        case DataType::VEC4_UINT32:
        case DataType::VEC4_INT32:
        case DataType::VEC4_UINT64:
        case DataType::VEC4_INT64:
        case DataType::VEC4_FLOAT32:
        case DataType::VEC4_FLOAT64:
        case DataType::VEC4_UINT8_NORM:
        case DataType::VEC4_INT8_NORM:
        case DataType::VEC4_UINT16_NORM:
        case DataType::VEC4_INT16_NORM:
        case DataType::VEC4_UINT32_NORM:
        case DataType::VEC4_INT32_NORM:
        case DataType::VEC4_UINT64_NORM:
        case DataType::VEC4_INT64_NORM:
            return TypeGroup::VECTOR;
        case DataType::MAT2_UINT8:
        case DataType::MAT2_INT8:
        case DataType::MAT2_UINT16:
        case DataType::MAT2_INT16:
        case DataType::MAT2_UINT32:
        case DataType::MAT2_INT32:
        case DataType::MAT2_UINT64:
        case DataType::MAT2_INT64:
        case DataType::MAT2_FLOAT32:
        case DataType::MAT2_FLOAT64:
        case DataType::MAT2_UINT8_NORM:
        case DataType::MAT2_INT8_NORM:
        case DataType::MAT2_UINT16_NORM:
        case DataType::MAT2_INT16_NORM:
        case DataType::MAT2_UINT32_NORM:
        case DataType::MAT2_INT32_NORM:
        case DataType::MAT2_UINT64_NORM:
        case DataType::MAT2_INT64_NORM:
        case DataType::MAT3_UINT8:
        case DataType::MAT3_INT8:
        case DataType::MAT3_UINT16:
        case DataType::MAT3_INT16:
        case DataType::MAT3_UINT32:
        case DataType::MAT3_INT32:
        case DataType::MAT3_UINT64:
        case DataType::MAT3_INT64:
        case DataType::MAT3_FLOAT32:
        case DataType::MAT3_FLOAT64:
        case DataType::MAT3_UINT8_NORM:
        case DataType::MAT3_INT8_NORM:
        case DataType::MAT3_UINT16_NORM:
        case DataType::MAT3_INT16_NORM:
        case DataType::MAT3_UINT32_NORM:
        case DataType::MAT3_INT32_NORM:
        case DataType::MAT3_UINT64_NORM:
        case DataType::MAT3_INT64_NORM:
        case DataType::MAT4_UINT8:
        case DataType::MAT4_INT8:
        case DataType::MAT4_UINT16:
        case DataType::MAT4_INT16:
        case DataType::MAT4_UINT32:
        case DataType::MAT4_INT32:
        case DataType::MAT4_UINT64:
        case DataType::MAT4_INT64:
        case DataType::MAT4_FLOAT32:
        case DataType::MAT4_FLOAT64:
        case DataType::MAT4_UINT8_NORM:
        case DataType::MAT4_INT8_NORM:
        case DataType::MAT4_UINT16_NORM:
        case DataType::MAT4_INT16_NORM:
        case DataType::MAT4_UINT32_NORM:
        case DataType::MAT4_INT32_NORM:
        case DataType::MAT4_UINT64_NORM:
        case DataType::MAT4_INT64_NORM:
            return TypeGroup::MATRIX;
        case DataType::UNKNOWN:
            return TypeGroup::UNKNOWN;
    }

    // Shouldn't reach here
    assert(false);
    return TypeGroup::UNKNOWN;
}

constexpr DataType compose(DataType componentType, TypeGroup group, uint64_t componentCount, bool normalized) {
    assert(getComponentCount(componentType) == 1);
    assert(group != TypeGroup::VECTOR || componentCount == 2 || componentCount == 3 || componentCount == 4);
    assert(group != TypeGroup::MATRIX || componentCount == 4 || componentCount == 9 || componentCount == 16);
    assert(group != TypeGroup::SCALAR || componentCount == 1);

    switch (group) {
        case TypeGroup::SCALAR:
            switch (componentType) {
                case DataType::UINT8:
                    return normalized ? DataType::UINT8_NORM : DataType::UINT8;
                case DataType::INT8:
                    return normalized ? DataType::INT8_NORM : DataType::INT8;
                case DataType::UINT16:
                    return normalized ? DataType::UINT16_NORM : DataType::UINT16;
                case DataType::INT16:
                    return normalized ? DataType::INT16_NORM : DataType::INT16;
                case DataType::UINT32:
                    return normalized ? DataType::UINT32_NORM : DataType::UINT32;
                case DataType::INT32:
                    return normalized ? DataType::INT32_NORM : DataType::INT32;
                case DataType::UINT64:
                    return normalized ? DataType::UINT64_NORM : DataType::UINT64;
                case DataType::INT64:
                    return normalized ? DataType::INT64_NORM : DataType::INT64;
                case DataType::FLOAT32:
                    return DataType::FLOAT32;
                case DataType::FLOAT64:
                    return DataType::FLOAT64;
                default:
                    // Shouldn't reach here
                    assert(false);
                    return DataType::UNKNOWN;
            }
        case TypeGroup::VECTOR:
            switch (componentCount) {
                case 2:
                    switch (componentType) {
                        case DataType::UINT8:
                            return normalized ? DataType::VEC2_UINT8_NORM : DataType::VEC2_UINT8;
                        case DataType::INT8:
                            return normalized ? DataType::VEC2_INT8_NORM : DataType::VEC2_INT8;
                        case DataType::UINT16:
                            return normalized ? DataType::VEC2_UINT16_NORM : DataType::VEC2_UINT16;
                        case DataType::INT16:
                            return normalized ? DataType::VEC2_INT16_NORM : DataType::VEC2_INT16;
                        case DataType::UINT32:
                            return normalized ? DataType::VEC2_UINT32_NORM : DataType::VEC2_UINT32;
                        case DataType::INT32:
                            return normalized ? DataType::VEC2_INT32_NORM : DataType::VEC2_INT32;
                        case DataType::UINT64:
                            return normalized ? DataType::VEC2_UINT64_NORM : DataType::VEC2_UINT64;
                        case DataType::INT64:
                            return normalized ? DataType::VEC2_INT64_NORM : DataType::VEC2_INT64;
                        case DataType::FLOAT32:
                            return DataType::VEC2_FLOAT32;
                        case DataType::FLOAT64:
                            return DataType::VEC2_FLOAT64;
                        default:
                            // Shouldn't reach here
                            assert(false);
                            return DataType::UNKNOWN;
                    }
                case 3:
                    switch (componentType) {
                        case DataType::UINT8:
                            return normalized ? DataType::VEC3_UINT8_NORM : DataType::VEC3_UINT8;
                        case DataType::INT8:
                            return normalized ? DataType::VEC3_INT8_NORM : DataType::VEC3_INT8;
                        case DataType::UINT16:
                            return normalized ? DataType::VEC3_UINT16_NORM : DataType::VEC3_UINT16;
                        case DataType::INT16:
                            return normalized ? DataType::VEC3_INT16_NORM : DataType::VEC3_INT16;
                        case DataType::UINT32:
                            return normalized ? DataType::VEC3_UINT32_NORM : DataType::VEC3_UINT32;
                        case DataType::INT32:
                            return normalized ? DataType::VEC3_INT32_NORM : DataType::VEC3_INT32;
                        case DataType::UINT64:
                            return normalized ? DataType::VEC3_UINT64_NORM : DataType::VEC3_UINT64;
                        case DataType::INT64:
                            return normalized ? DataType::VEC3_INT64_NORM : DataType::VEC3_INT64;
                        case DataType::FLOAT32:
                            return DataType::VEC3_FLOAT32;
                        case DataType::FLOAT64:
                            return DataType::VEC3_FLOAT64;
                        default:
                            // Shouldn't reach here
                            assert(false);
                            return DataType::UNKNOWN;
                    }
                case 4:
                    switch (componentType) {
                        case DataType::UINT8:
                            return normalized ? DataType::VEC4_UINT8_NORM : DataType::VEC4_UINT8;
                        case DataType::INT8:
                            return normalized ? DataType::VEC4_INT8_NORM : DataType::VEC4_INT8;
                        case DataType::UINT16:
                            return normalized ? DataType::VEC4_UINT16_NORM : DataType::VEC4_UINT16;
                        case DataType::INT16:
                            return normalized ? DataType::VEC4_INT16_NORM : DataType::VEC4_INT16;
                        case DataType::UINT32:
                            return normalized ? DataType::VEC4_UINT32_NORM : DataType::VEC4_UINT32;
                        case DataType::INT32:
                            return normalized ? DataType::VEC4_INT32_NORM : DataType::VEC4_INT32;
                        case DataType::UINT64:
                            return normalized ? DataType::VEC4_UINT64_NORM : DataType::VEC4_UINT64;
                        case DataType::INT64:
                            return normalized ? DataType::VEC4_INT64_NORM : DataType::VEC4_INT64;
                        case DataType::FLOAT32:
                            return DataType::VEC4_FLOAT32;
                        case DataType::FLOAT64:
                            return DataType::VEC4_FLOAT64;
                        default:
                            // Shouldn't reach here
                            assert(false);
                            return DataType::UNKNOWN;
                    }
                default:
                    // Shouldn't reach here
                    assert(false);
                    return DataType::UNKNOWN;
            }
        case TypeGroup::MATRIX:
            switch (componentCount) {
                case 4:
                    switch (componentType) {
                        case DataType::UINT8:
                            return normalized ? DataType::MAT2_UINT8_NORM : DataType::MAT2_UINT8;
                        case DataType::INT8:
                            return normalized ? DataType::MAT2_INT8_NORM : DataType::MAT2_INT8;
                        case DataType::UINT16:
                            return normalized ? DataType::MAT2_UINT16_NORM : DataType::MAT2_UINT16;
                        case DataType::INT16:
                            return normalized ? DataType::MAT2_INT16_NORM : DataType::MAT2_INT16;
                        case DataType::UINT32:
                            return normalized ? DataType::MAT2_UINT32_NORM : DataType::MAT2_UINT32;
                        case DataType::INT32:
                            return normalized ? DataType::MAT2_INT32_NORM : DataType::MAT2_INT32;
                        case DataType::UINT64:
                            return normalized ? DataType::MAT2_UINT64_NORM : DataType::MAT2_UINT64;
                        case DataType::INT64:
                            return normalized ? DataType::MAT2_INT64_NORM : DataType::MAT2_INT64;
                        case DataType::FLOAT32:
                            return DataType::MAT2_FLOAT32;
                        case DataType::FLOAT64:
                            return DataType::MAT2_FLOAT64;
                        default:
                            // Shouldn't reach here
                            assert(false);
                            return DataType::UNKNOWN;
                    }
                case 9:
                    switch (componentType) {
                        case DataType::UINT8:
                            return normalized ? DataType::MAT3_UINT8_NORM : DataType::MAT3_UINT8;
                        case DataType::INT8:
                            return normalized ? DataType::MAT3_INT8_NORM : DataType::MAT3_INT8;
                        case DataType::UINT16:
                            return normalized ? DataType::MAT3_UINT16_NORM : DataType::MAT3_UINT16;
                        case DataType::INT16:
                            return normalized ? DataType::MAT3_INT16_NORM : DataType::MAT3_INT16;
                        case DataType::UINT32:
                            return normalized ? DataType::MAT3_UINT32_NORM : DataType::MAT3_UINT32;
                        case DataType::INT32:
                            return normalized ? DataType::MAT3_INT32_NORM : DataType::MAT3_INT32;
                        case DataType::UINT64:
                            return normalized ? DataType::MAT3_UINT64_NORM : DataType::MAT3_UINT64;
                        case DataType::INT64:
                            return normalized ? DataType::MAT3_INT64_NORM : DataType::MAT3_INT64;
                        case DataType::FLOAT32:
                            return DataType::MAT3_FLOAT32;
                        case DataType::FLOAT64:
                            return DataType::MAT3_FLOAT64;
                        default:
                            // Shouldn't reach here
                            assert(false);
                            return DataType::UNKNOWN;
                    }
                case 16:
                    switch (componentType) {
                        case DataType::UINT8:
                            return normalized ? DataType::MAT4_UINT8_NORM : DataType::MAT4_UINT8;
                        case DataType::INT8:
                            return normalized ? DataType::MAT4_INT8_NORM : DataType::MAT4_INT8;
                        case DataType::UINT16:
                            return normalized ? DataType::MAT4_UINT16_NORM : DataType::MAT4_UINT16;
                        case DataType::INT16:
                            return normalized ? DataType::MAT4_INT16_NORM : DataType::MAT4_INT16;
                        case DataType::UINT32:
                            return normalized ? DataType::MAT4_UINT32_NORM : DataType::MAT4_UINT32;
                        case DataType::INT32:
                            return normalized ? DataType::MAT4_INT32_NORM : DataType::MAT4_INT32;
                        case DataType::UINT64:
                            return normalized ? DataType::MAT4_UINT64_NORM : DataType::MAT4_UINT64;
                        case DataType::INT64:
                            return normalized ? DataType::MAT4_INT64_NORM : DataType::MAT4_INT64;
                        case DataType::FLOAT32:
                            return DataType::MAT4_FLOAT32;
                        case DataType::FLOAT64:
                            return DataType::MAT4_FLOAT64;
                        default:
                            // Shouldn't reach here
                            assert(false);
                            return DataType::UNKNOWN;
                    }
                default:
                    // Shouldn't reach here
                    assert(false);
                    return DataType::UNKNOWN;
            }
        case TypeGroup::UNKNOWN:
            // Shouldn't reach here
            assert(false);
            return DataType::UNKNOWN;
    }

    // Shouldn't reach here
    assert(false);
    return DataType::UNKNOWN;
}

constexpr bool isPrimvarType(DataType type) {
    // clang-format off
    switch (type) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::FLOAT32:
        case DataType::UINT8_NORM:
        case DataType::INT8_NORM:
        case DataType::UINT16_NORM:
        case DataType::INT16_NORM:
        case DataType::VEC2_UINT8:
        case DataType::VEC2_INT8:
        case DataType::VEC2_UINT16:
        case DataType::VEC2_INT16:
        case DataType::VEC2_FLOAT32:
        case DataType::VEC2_UINT8_NORM:
        case DataType::VEC2_INT8_NORM:
        case DataType::VEC2_UINT16_NORM:
        case DataType::VEC2_INT16_NORM:
        case DataType::VEC3_UINT8:
        case DataType::VEC3_INT8:
        case DataType::VEC3_UINT16:
        case DataType::VEC3_INT16:
        case DataType::VEC3_FLOAT32:
        case DataType::VEC4_FLOAT32:
        case DataType::VEC3_UINT8_NORM:
        case DataType::VEC3_INT8_NORM:
        case DataType::VEC3_UINT16_NORM:
        case DataType::VEC3_INT16_NORM:
        case DataType::VEC4_UINT8:
        case DataType::VEC4_INT8:
        case DataType::VEC4_UINT16:
        case DataType::VEC4_INT16:
        case DataType::VEC4_UINT8_NORM:
        case DataType::VEC4_INT8_NORM:
        case DataType::VEC4_UINT16_NORM:
        case DataType::VEC4_INT16_NORM:
            return true;
        default:
            return false;
    }
    // clang-format on
}

constexpr DataType getMdlPropertyType(DataType type) {
    // clang-format off
    switch (type) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT64:
        case DataType::INT64:
            return DataType::INT32;
        case DataType::FLOAT32:
        case DataType::FLOAT64:
        case DataType::UINT8_NORM:
        case DataType::INT8_NORM:
        case DataType::UINT16_NORM:
        case DataType::INT16_NORM:
        case DataType::UINT32_NORM:
        case DataType::INT32_NORM:
        case DataType::UINT64_NORM:
        case DataType::INT64_NORM:
            return DataType::FLOAT32;
        case DataType::VEC2_UINT8:
        case DataType::VEC2_INT8:
        case DataType::VEC2_UINT16:
        case DataType::VEC2_INT16:
        case DataType::VEC2_UINT32:
        case DataType::VEC2_INT32:
        case DataType::VEC2_UINT64:
        case DataType::VEC2_INT64:
            return DataType::VEC2_INT32;
        case DataType::VEC2_FLOAT32:
        case DataType::VEC2_FLOAT64:
        case DataType::VEC2_UINT8_NORM:
        case DataType::VEC2_INT8_NORM:
        case DataType::VEC2_UINT16_NORM:
        case DataType::VEC2_INT16_NORM:
        case DataType::VEC2_UINT32_NORM:
        case DataType::VEC2_INT32_NORM:
        case DataType::VEC2_UINT64_NORM:
        case DataType::VEC2_INT64_NORM:
            return DataType::VEC2_FLOAT32;
        case DataType::VEC3_UINT8:
        case DataType::VEC3_INT8:
        case DataType::VEC3_UINT16:
        case DataType::VEC3_INT16:
        case DataType::VEC3_UINT32:
        case DataType::VEC3_INT32:
        case DataType::VEC3_UINT64:
        case DataType::VEC3_INT64:
            return DataType::VEC3_INT32;
        case DataType::VEC3_FLOAT32:
        case DataType::VEC3_FLOAT64:
        case DataType::VEC3_UINT8_NORM:
        case DataType::VEC3_INT8_NORM:
        case DataType::VEC3_UINT16_NORM:
        case DataType::VEC3_INT16_NORM:
        case DataType::VEC3_UINT32_NORM:
        case DataType::VEC3_INT32_NORM:
        case DataType::VEC3_UINT64_NORM:
        case DataType::VEC3_INT64_NORM:
            return DataType::VEC3_FLOAT32;
        case DataType::VEC4_UINT8:
        case DataType::VEC4_INT8:
        case DataType::VEC4_UINT16:
        case DataType::VEC4_INT16:
        case DataType::VEC4_UINT32:
        case DataType::VEC4_INT32:
        case DataType::VEC4_UINT64:
        case DataType::VEC4_INT64:
            return DataType::VEC4_INT32;
        case DataType::VEC4_FLOAT32:
        case DataType::VEC4_FLOAT64:
        case DataType::VEC4_UINT8_NORM:
        case DataType::VEC4_INT8_NORM:
        case DataType::VEC4_UINT16_NORM:
        case DataType::VEC4_INT16_NORM:
        case DataType::VEC4_UINT32_NORM:
        case DataType::VEC4_INT32_NORM:
        case DataType::VEC4_UINT64_NORM:
        case DataType::VEC4_INT64_NORM:
            return DataType::VEC4_FLOAT32;
        case DataType::MAT2_UINT8:
        case DataType::MAT2_INT8:
        case DataType::MAT2_UINT16:
        case DataType::MAT2_INT16:
        case DataType::MAT2_UINT32:
        case DataType::MAT2_INT32:
        case DataType::MAT2_UINT64:
        case DataType::MAT2_INT64:
        case DataType::MAT2_FLOAT32:
        case DataType::MAT2_FLOAT64:
        case DataType::MAT2_UINT8_NORM:
        case DataType::MAT2_INT8_NORM:
        case DataType::MAT2_UINT16_NORM:
        case DataType::MAT2_INT16_NORM:
        case DataType::MAT2_UINT32_NORM:
        case DataType::MAT2_INT32_NORM:
        case DataType::MAT2_UINT64_NORM:
        case DataType::MAT2_INT64_NORM:
        case DataType::MAT3_UINT8:
        case DataType::MAT3_INT8:
        case DataType::MAT3_UINT16:
        case DataType::MAT3_INT16:
        case DataType::MAT3_UINT32:
        case DataType::MAT3_INT32:
        case DataType::MAT3_UINT64:
        case DataType::MAT3_INT64:
        case DataType::MAT3_FLOAT32:
        case DataType::MAT3_FLOAT64:
        case DataType::MAT3_UINT8_NORM:
        case DataType::MAT3_INT8_NORM:
        case DataType::MAT3_UINT16_NORM:
        case DataType::MAT3_INT16_NORM:
        case DataType::MAT3_UINT32_NORM:
        case DataType::MAT3_INT32_NORM:
        case DataType::MAT3_UINT64_NORM:
        case DataType::MAT3_INT64_NORM:
        case DataType::MAT4_UINT8:
        case DataType::MAT4_INT8:
        case DataType::MAT4_UINT16:
        case DataType::MAT4_INT16:
        case DataType::MAT4_UINT32:
        case DataType::MAT4_INT32:
        case DataType::MAT4_UINT64:
        case DataType::MAT4_INT64:
        case DataType::MAT4_FLOAT32:
        case DataType::MAT4_FLOAT64:
        case DataType::MAT4_UINT8_NORM:
        case DataType::MAT4_INT8_NORM:
        case DataType::MAT4_UINT16_NORM:
        case DataType::MAT4_INT16_NORM:
        case DataType::MAT4_UINT32_NORM:
        case DataType::MAT4_INT32_NORM:
        case DataType::MAT4_UINT64_NORM:
        case DataType::MAT4_INT64_NORM:
        case DataType::UNKNOWN:
            return DataType::UNKNOWN;
    }
    // clang-format on

    // Unreachable code. All enum cases are handled above.
    assert(false);
    return DataType::UNKNOWN;
}

inline DataType getGltfVertexAttributeType(const std::string& type, int32_t gltfComponentType, bool normalized) {
    auto componentType = DataType::UNKNOWN;

    if (gltfComponentType == CesiumGltf::Accessor::ComponentType::BYTE) {
        componentType = DataType::INT8;
    } else if (gltfComponentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
        componentType = DataType::UINT8;
    } else if (gltfComponentType == CesiumGltf::Accessor::ComponentType::SHORT) {
        componentType = DataType::INT16;
    } else if (gltfComponentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
        componentType = DataType::UINT16;
    } else if (gltfComponentType == CesiumGltf::Accessor::ComponentType::FLOAT) {
        componentType = DataType::FLOAT32;
    }

    if (componentType == DataType::UNKNOWN) {
        return DataType::UNKNOWN;
    }

    if (type == CesiumGltf::Accessor::Type::SCALAR) {
        return compose(componentType, TypeGroup::SCALAR, 1, normalized);
    } else if (type == CesiumGltf::Accessor::Type::VEC2) {
        return compose(componentType, TypeGroup::VECTOR, 2, normalized);
    } else if (type == CesiumGltf::Accessor::Type::VEC3) {
        return compose(componentType, TypeGroup::VECTOR, 3, normalized);
    } else if (type == CesiumGltf::Accessor::Type::VEC4) {
        return compose(componentType, TypeGroup::VECTOR, 4, normalized);
    } else if (type == CesiumGltf::Accessor::Type::MAT2) {
        return compose(componentType, TypeGroup::MATRIX, 4, normalized);
    } else if (type == CesiumGltf::Accessor::Type::MAT3) {
        return compose(componentType, TypeGroup::MATRIX, 9, normalized);
    } else if (type == CesiumGltf::Accessor::Type::MAT4) {
        return compose(componentType, TypeGroup::MATRIX, 16, normalized);
    }

    return DataType::UNKNOWN;
}

inline DataType getClassPropertyType(const CesiumGltf::Schema& schema, const CesiumGltf::ClassProperty& classProperty) {
    const auto isArray = classProperty.array;
    const auto arrayCount = static_cast<uint64_t>(classProperty.count.value_or(0));
    const auto arrayCountInRange = arrayCount > 1 && arrayCount <= 4;

    auto propertyType = CesiumGltf::convertStringToPropertyType(classProperty.type);
    auto propertyComponentType =
        CesiumGltf::convertStringToPropertyComponentType(classProperty.componentType.value_or(std::string{}));

    const auto isEnum = propertyType == CesiumGltf::PropertyType::Enum;

    if (isEnum) {
        const auto& enumType = classProperty.enumType;
        if (!enumType.has_value()) {
            return DataType::UNKNOWN;
        }
        const auto& enumIter = schema.enums.find(enumType.value());
        if (enumIter == schema.enums.end()) {
            return DataType::UNKNOWN;
        }
        const auto& enumDefinition = enumIter->second;

        // Treat enum as scalar
        propertyType = CesiumGltf::PropertyType::Scalar;
        propertyComponentType = CesiumGltf::convertStringToPropertyComponentType(enumDefinition.valueType);
    }

    const auto isScalar = propertyType == CesiumGltf::PropertyType::Scalar;
    const auto isVector = CesiumGltf::isPropertyTypeVecN(propertyType);
    const auto isMatrix = CesiumGltf::isPropertyTypeMatN(propertyType);
    const auto normalized = classProperty.normalized;

    if (!isScalar && !isVector && !isMatrix) {
        // Only scalars, vectors, and matrices are supported
        return DataType::UNKNOWN;
    }

    if (isArray && (!isScalar || !arrayCountInRange)) {
        // Only arrays with 2, 3, or 4 scalar elements are supported
        return DataType::UNKNOWN;
    }

    if (propertyType == CesiumGltf::PropertyType::Invalid) {
        // Something went wrong or invalid schema
        return DataType::UNKNOWN;
    }

    if (propertyComponentType == CesiumGltf::PropertyComponentType::None) {
        // Something went wrong or invalid schema
        return DataType::UNKNOWN;
    }

    auto componentType = DataType::UNKNOWN;

    switch (propertyComponentType) {
        case CesiumGltf::PropertyComponentType::Int8:
            componentType = DataType::INT8;
            break;
        case CesiumGltf::PropertyComponentType::Uint8:
            componentType = DataType::UINT8;
            break;
        case CesiumGltf::PropertyComponentType::Int16:
            componentType = DataType::INT16;
            break;
        case CesiumGltf::PropertyComponentType::Uint16:
            componentType = DataType::UINT16;
            break;
        case CesiumGltf::PropertyComponentType::Int32:
            componentType = DataType::INT32;
            break;
        case CesiumGltf::PropertyComponentType::Uint32:
            componentType = DataType::UINT32;
            break;
        case CesiumGltf::PropertyComponentType::Int64:
            componentType = DataType::INT64;
            break;
        case CesiumGltf::PropertyComponentType::Uint64:
            componentType = DataType::UINT64;
            break;
        case CesiumGltf::PropertyComponentType::Float32:
            componentType = DataType::FLOAT32;
            break;
        case CesiumGltf::PropertyComponentType::Float64:
            componentType = DataType::FLOAT64;
            break;
        default:
            break;
    }

    if (componentType == DataType::UNKNOWN) {
        // Shouldn't ever reach here
        return DataType::UNKNOWN;
    }

    if (isArray) {
        return compose(componentType, TypeGroup::SCALAR, arrayCount, normalized);
    }

    switch (propertyType) {
        case CesiumGltf::PropertyType::Scalar:
            return compose(componentType, TypeGroup::SCALAR, 1, normalized);
        case CesiumGltf::PropertyType::Vec2:
            return compose(componentType, TypeGroup::VECTOR, 2, normalized);
        case CesiumGltf::PropertyType::Vec3:
            return compose(componentType, TypeGroup::VECTOR, 3, normalized);
        case CesiumGltf::PropertyType::Vec4:
            return compose(componentType, TypeGroup::VECTOR, 4, normalized);
        case CesiumGltf::PropertyType::Mat2:
            return compose(componentType, TypeGroup::MATRIX, 4, normalized);
        case CesiumGltf::PropertyType::Mat3:
            return compose(componentType, TypeGroup::MATRIX, 9, normalized);
        case CesiumGltf::PropertyType::Mat4:
            return compose(componentType, TypeGroup::MATRIX, 16, normalized);
        default:
            break;
    }

    // Shouldn't ever reach here
    return DataType::UNKNOWN;
}

constexpr DataType getPrimvarType(DataType type) {
    assert(isPrimvarType(type));

    // clang-format off
    switch (type) {
        case DataType::UINT8: return GetNativeTypeReverse<GetPrimvarType<DataType::UINT8>>::Type;
        case DataType::INT8: return GetNativeTypeReverse<GetPrimvarType<DataType::INT8>>::Type;
        case DataType::UINT16: return GetNativeTypeReverse<GetPrimvarType<DataType::UINT16>>::Type;
        case DataType::INT16: return GetNativeTypeReverse<GetPrimvarType<DataType::INT16>>::Type;
        case DataType::FLOAT32: return GetNativeTypeReverse<GetPrimvarType<DataType::FLOAT32>>::Type;
        case DataType::UINT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::UINT8_NORM>>::Type;
        case DataType::INT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::INT8_NORM>>::Type;
        case DataType::UINT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::UINT16_NORM>>::Type;
        case DataType::INT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::INT16_NORM>>::Type;
        case DataType::VEC2_UINT8: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_UINT8>>::Type;
        case DataType::VEC2_INT8: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_INT8>>::Type;
        case DataType::VEC2_UINT16:  return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_UINT16>>::Type;
        case DataType::VEC2_INT16: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_INT16>>::Type;
        case DataType::VEC2_FLOAT32: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_FLOAT32>>::Type;
        case DataType::VEC2_UINT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_UINT8_NORM>>::Type;
        case DataType::VEC2_INT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_INT8_NORM>>::Type;
        case DataType::VEC2_UINT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_UINT16_NORM>>::Type;
        case DataType::VEC2_INT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC2_INT16_NORM>>::Type;
        case DataType::VEC3_UINT8: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_UINT8>>::Type;
        case DataType::VEC3_INT8: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_INT8>>::Type;
        case DataType::VEC3_UINT16: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_UINT16>>::Type;
        case DataType::VEC3_INT16: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_INT16>>::Type;
        case DataType::VEC3_FLOAT32: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_FLOAT32>>::Type;
        case DataType::VEC3_UINT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_UINT8_NORM>>::Type;
        case DataType::VEC3_INT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_INT8_NORM>>::Type;
        case DataType::VEC3_UINT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_UINT16_NORM>>::Type;
        case DataType::VEC3_INT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC3_INT16_NORM>>::Type;
        case DataType::VEC4_UINT8: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_UINT8>>::Type;
        case DataType::VEC4_INT8: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_INT8>>::Type;
        case DataType::VEC4_UINT16: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_UINT16>>::Type;
        case DataType::VEC4_INT16: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_INT16>>::Type;
        case DataType::VEC4_FLOAT32: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_FLOAT32>>::Type;
        case DataType::VEC4_UINT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_UINT8_NORM>>::Type;
        case DataType::VEC4_INT8_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_INT8_NORM>>::Type;
        case DataType::VEC4_UINT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_UINT16_NORM>>::Type;
        case DataType::VEC4_INT16_NORM: return GetNativeTypeReverse<GetPrimvarType<DataType::VEC4_INT16_NORM>>::Type;
        default:
            // Not a valid vertex attribute type
            assert(false);
            return DataType::UNKNOWN;
    }
    // clang-format on
}

constexpr omni::fabric::BaseDataType getPrimvarBaseDataType(DataType type) {
    assert(isPrimvarType(type));

    const auto componentType = getComponentType(getPrimvarType(type));
    switch (componentType) {
        case DataType::UINT8:
            return omni::fabric::BaseDataType::eUChar;
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
            return omni::fabric::BaseDataType::eInt;
        case DataType::FLOAT32:
            return omni::fabric::BaseDataType::eFloat;
        default:
            // Not a valid vertex attribute type
            assert(false);
            return omni::fabric::BaseDataType::eUnknown;
    }
}

constexpr omni::fabric::Type getFabricPrimvarType(DataType type) {
    assert(isPrimvarType(type));
    const auto baseDataType = getPrimvarBaseDataType(type);
    const auto componentCount = getComponentCount(type);
    return {baseDataType, static_cast<uint8_t>(componentCount), 1, omni::fabric::AttributeRole::eNone};
}

} // namespace cesium::omniverse
