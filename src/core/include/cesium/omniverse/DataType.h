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

template <DataType T> struct IsFloatingPoint;
template <> struct IsFloatingPoint<DataType::UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::INT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::VEC2_FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC2_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::VEC3_FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC3_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::VEC4_FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::VEC4_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::MAT2_FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT2_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::MAT3_FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT3_INT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT8> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT16> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT32> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT64> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_FLOAT32> : std::true_type {};
template <> struct IsFloatingPoint<DataType::MAT4_FLOAT64> : std::true_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT8_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT16_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT32_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_UINT64_NORM> : std::false_type {};
template <> struct IsFloatingPoint<DataType::MAT4_INT64_NORM> : std::false_type {};

template <DataType T> struct IsMatrix;
template <> struct IsMatrix<DataType::UINT8> : std::false_type {};
template <> struct IsMatrix<DataType::INT8> : std::false_type {};
template <> struct IsMatrix<DataType::UINT16> : std::false_type {};
template <> struct IsMatrix<DataType::INT16> : std::false_type {};
template <> struct IsMatrix<DataType::UINT32> : std::false_type {};
template <> struct IsMatrix<DataType::INT32> : std::false_type {};
template <> struct IsMatrix<DataType::UINT64> : std::false_type {};
template <> struct IsMatrix<DataType::INT64> : std::false_type {};
template <> struct IsMatrix<DataType::FLOAT32> : std::false_type {};
template <> struct IsMatrix<DataType::FLOAT64> : std::false_type {};
template <> struct IsMatrix<DataType::UINT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::INT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::UINT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::INT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::UINT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::INT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::UINT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::INT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT8> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT8> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT16> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT16> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_FLOAT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_FLOAT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_UINT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC2_INT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT8> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT8> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT16> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT16> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_FLOAT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_FLOAT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_UINT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC3_INT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT8> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT8> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT16> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT16> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_FLOAT32> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_FLOAT64> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT8_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT16_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT32_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_UINT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::VEC4_INT64_NORM> : std::false_type {};
template <> struct IsMatrix<DataType::MAT2_UINT8> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT8> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT16> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT16> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_FLOAT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_FLOAT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT8_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT8_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT16_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT16_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT32_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT32_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_UINT64_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT2_INT64_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT8> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT8> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT16> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT16> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_FLOAT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_FLOAT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT8_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT8_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT16_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT16_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT32_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT32_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_UINT64_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT3_INT64_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT8> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT8> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT16> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT16> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_FLOAT32> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_FLOAT64> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT8_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT8_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT16_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT16_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT32_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT32_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_UINT64_NORM> : std::true_type {};
template <> struct IsMatrix<DataType::MAT4_INT64_NORM> : std::true_type {};

// GLM does not typedef all integer matrix types, so they are written in full.
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
template <> struct GetRawTypeImpl<DataType::MAT2_UINT8> { using Type = glm::mat<2, 2, uint8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT8> { using Type = glm::mat<2, 2, int8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT16> { using Type = glm::mat<2, 2, uint16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT16> { using Type = glm::mat<2, 2, int16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT32> { using Type = glm::mat<2, 2, uint32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT32> { using Type = glm::mat<2, 2, int32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT64> { using Type = glm::mat<2, 2, uint64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT64> { using Type = glm::mat<2, 2, int64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f64mat2; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::mat<2, 2, uint8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::mat<2, 2, int8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::mat<2, 2, uint16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::mat<2, 2, int16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::mat<2, 2, uint32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::mat<2, 2, int32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::mat<2, 2, uint64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::mat<2, 2, int64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT8> { using Type = glm::mat<3, 3, uint8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT8> { using Type = glm::mat<3, 3, int8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT16> { using Type = glm::mat<3, 3, uint16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT16> { using Type = glm::mat<3, 3, int16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT32> { using Type = glm::mat<3, 3, uint32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT32> { using Type = glm::mat<3, 3, int32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT64> { using Type = glm::mat<3, 3, uint64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT64> { using Type = glm::mat<3, 3, int64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f64mat3; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::mat<3, 3, uint8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::mat<3, 3, int8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::mat<3, 3, uint16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::mat<3, 3, int16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::mat<3, 3, uint32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::mat<3, 3, int32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::mat<3, 3, uint64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::mat<3, 3, int64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT8> { using Type = glm::mat<4, 4, uint8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT8> { using Type = glm::mat<4, 4, int8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT16> { using Type = glm::mat<4, 4, uint16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT16> { using Type = glm::mat<4, 4, int16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT32> { using Type = glm::mat<4, 4, uint32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT32> { using Type = glm::mat<4, 4, int32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT64> { using Type = glm::mat<4, 4, uint64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT64> { using Type = glm::mat<4, 4, int64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f64mat4; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::mat<4, 4, uint8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::mat<4, 4, int8_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::mat<4, 4, uint16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::mat<4, 4, int16_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::mat<4, 4, uint32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::mat<4, 4, int32_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::mat<4, 4, uint64_t>; };
template <> struct GetRawTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::mat<4, 4, int64_t>; };

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
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT8> { using Type = glm::mat<2, 2, uint8_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT8> { using Type = glm::mat<2, 2, int8_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT16> { using Type = glm::mat<2, 2, uint16_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT16> { using Type = glm::mat<2, 2, int16_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT32> { using Type = glm::mat<2, 2, uint32_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT32> { using Type = glm::mat<2, 2, int32_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_UINT64> { using Type = glm::mat<2, 2, uint64_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT2_INT64> { using Type = glm::mat<2, 2, int64_t>; };
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
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT8> { using Type = glm::mat<3, 3, uint8_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT8> { using Type = glm::mat<3, 3, int8_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT16> { using Type = glm::mat<3, 3, uint16_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT16> { using Type = glm::mat<3, 3, int16_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT32> { using Type = glm::mat<3, 3, uint32_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT32> { using Type = glm::mat<3, 3, int32_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_UINT64> { using Type = glm::mat<3, 3, uint64_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT3_INT64> { using Type = glm::mat<3, 3, int64_t>; };
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
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT8> { using Type = glm::mat<4, 4, uint8_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT8> { using Type = glm::mat<4, 4, int8_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT16> { using Type = glm::mat<4, 4, uint16_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT16> { using Type = glm::mat<4, 4, int16_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT32> { using Type = glm::mat<4, 4, uint32_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT32> { using Type = glm::mat<4, 4, int32_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_UINT64> { using Type = glm::mat<4, 4, uint64_t>; };
template <> struct GetTransformedTypeImpl<DataType::MAT4_INT64> { using Type = glm::mat<4, 4, int64_t>; };
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

template <DataType T> struct GetMdlRawTypeImpl;
template <> struct GetMdlRawTypeImpl<DataType::UINT8> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT8> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::UINT16> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT16> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::UINT32> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT32> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::UINT64> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT64> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetMdlRawTypeImpl<DataType::FLOAT64> { using Type = float; };
template <> struct GetMdlRawTypeImpl<DataType::UINT8_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT8_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::UINT16_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT16_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::UINT32_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT32_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::UINT64_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::INT64_NORM> { using Type = int; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT8> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT8> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT16> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT16> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT32> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT64> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT64> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_FLOAT64> { using Type = glm::f32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT32_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC2_INT64_NORM> { using Type = glm::i32vec2; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT8> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT8> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT16> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT16> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT32> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT64> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT64> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_FLOAT64> { using Type = glm::f32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT32_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC3_INT64_NORM> { using Type = glm::i32vec3; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT8> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT8> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT16> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT16> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT32> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT64> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT64> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_FLOAT64> { using Type = glm::f32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT32_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::VEC4_INT64_NORM> { using Type = glm::i32vec4; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT8> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT8> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT16> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT16> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT32> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT32> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT64> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT64> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f32mat2; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT8> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT8> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT16> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT16> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT32> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT32> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT64> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT64> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f32mat3; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT8> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT8> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT16> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT16> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT32> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT32> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT64> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT64> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f32mat4; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlRawTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::mat<4, 4, int>; };

template <DataType T> using GetMdlRawType = typename GetMdlRawTypeImpl<T>::Type;

template <DataType T> struct GetMdlTransformedTypeImpl;
template <> struct GetMdlTransformedTypeImpl<DataType::UINT8> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT8> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT16> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT16> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT32> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT32> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT64> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT64> { using Type = int; };
template <> struct GetMdlTransformedTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::FLOAT64> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT8_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT8_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT16_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT16_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT32_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT32_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::UINT64_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::INT64_NORM> { using Type = float; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT8> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT8> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT16> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT16> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT32> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT32> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT64> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT64> { using Type = glm::i32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_FLOAT64> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT32_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT32_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_UINT64_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC2_INT64_NORM> { using Type = glm::f32vec2; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT8> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT8> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT16> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT16> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT32> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT32> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT64> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT64> { using Type = glm::i32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_FLOAT64> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT32_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT32_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_UINT64_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC3_INT64_NORM> { using Type = glm::f32vec3; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT8> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT8> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT16> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT16> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT32> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT32> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT64> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT64> { using Type = glm::i32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_FLOAT64> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT32_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT32_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_UINT64_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::VEC4_INT64_NORM> { using Type = glm::f32vec4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT8> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT8> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT16> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT16> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT32> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT32> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT64> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT64> { using Type = glm::mat<2, 2, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_FLOAT32> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_FLOAT64> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT8_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT8_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT16_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT16_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT32_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT32_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_UINT64_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT2_INT64_NORM> { using Type = glm::f32mat2; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT8> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT8> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT16> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT16> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT32> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT32> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT64> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT64> { using Type = glm::mat<3, 3, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_FLOAT32> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_FLOAT64> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT8_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT8_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT16_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT16_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT32_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT32_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_UINT64_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT3_INT64_NORM> { using Type = glm::f32mat3; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT8> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT8> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT16> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT16> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT32> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT32> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT64> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT64> { using Type = glm::mat<4, 4, int>; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_FLOAT32> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_FLOAT64> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT8_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT8_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT16_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT16_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT32_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT32_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_UINT64_NORM> { using Type = glm::f32mat4; };
template <> struct GetMdlTransformedTypeImpl<DataType::MAT4_INT64_NORM> { using Type = glm::f32mat4; };

template <DataType T> using GetMdlTransformedType = typename GetMdlTransformedTypeImpl<T>::Type;

template <DataType T> struct GetMdlShaderType;

// clang-format off
template <> struct GetMdlShaderType<DataType::UINT8> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::INT8> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::UINT16> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::INT16> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::UINT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::INT32> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::UINT64> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::INT64> { static constexpr auto Type = DataType::INT32;};
template <> struct GetMdlShaderType<DataType::FLOAT32> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetMdlShaderType<DataType::FLOAT64> { static constexpr auto Type = DataType::FLOAT32;};
template <> struct GetMdlShaderType<DataType::UINT8_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::INT8_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::UINT16_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::INT16_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::UINT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::INT32_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::UINT64_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::INT64_NORM> { static constexpr auto Type = DataType::INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT8> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_INT8> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT16> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_INT16> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT32> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_INT32> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT64> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_INT64> { static constexpr auto Type = DataType::VEC2_INT32;};
template <> struct GetMdlShaderType<DataType::VEC2_FLOAT32> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetMdlShaderType<DataType::VEC2_FLOAT64> { static constexpr auto Type = DataType::VEC2_FLOAT32;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT8_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_INT8_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT16_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_INT16_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT32_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_INT32_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_UINT64_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC2_INT64_NORM> { static constexpr auto Type = DataType::VEC2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT8> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_INT8> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT16> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_INT16> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT32> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_INT32> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT64> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_INT64> { static constexpr auto Type = DataType::VEC3_INT32;};
template <> struct GetMdlShaderType<DataType::VEC3_FLOAT32> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetMdlShaderType<DataType::VEC3_FLOAT64> { static constexpr auto Type = DataType::VEC3_FLOAT32;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT8_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_INT8_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT16_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_INT16_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT32_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_INT32_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_UINT64_NORM> { static constexpr auto Type = DataType::VEC3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC3_INT64_NORM> { static constexpr auto Type = DataType::VEC3_INT64_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT8> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_INT8> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT16> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_INT16> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT32> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_INT32> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT64> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_INT64> { static constexpr auto Type = DataType::VEC4_INT32;};
template <> struct GetMdlShaderType<DataType::VEC4_FLOAT32> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetMdlShaderType<DataType::VEC4_FLOAT64> { static constexpr auto Type = DataType::VEC4_FLOAT32;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT8_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_INT8_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT16_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_INT16_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT32_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_INT32_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_UINT64_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::VEC4_INT64_NORM> { static constexpr auto Type = DataType::VEC4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT8> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_INT8> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT16> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_INT16> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT32> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_INT32> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT64> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_INT64> { static constexpr auto Type = DataType::MAT2_INT32;};
template <> struct GetMdlShaderType<DataType::MAT2_FLOAT32> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetMdlShaderType<DataType::MAT2_FLOAT64> { static constexpr auto Type = DataType::MAT2_FLOAT32;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT8_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_INT8_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT16_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_INT16_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT32_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_INT32_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_UINT64_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT2_INT64_NORM> { static constexpr auto Type = DataType::MAT2_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT8> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_INT8> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT16> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_INT16> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT32> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_INT32> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT64> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_INT64> { static constexpr auto Type = DataType::MAT3_INT32;};
template <> struct GetMdlShaderType<DataType::MAT3_FLOAT32> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetMdlShaderType<DataType::MAT3_FLOAT64> { static constexpr auto Type = DataType::MAT3_FLOAT32;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT8_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_INT8_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT16_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_INT16_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT32_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_INT32_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_UINT64_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT3_INT64_NORM> { static constexpr auto Type = DataType::MAT3_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT8> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_INT8> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT16> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_INT16> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT32> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_INT32> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT64> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_INT64> { static constexpr auto Type = DataType::MAT4_INT32;};
template <> struct GetMdlShaderType<DataType::MAT4_FLOAT32> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetMdlShaderType<DataType::MAT4_FLOAT64> { static constexpr auto Type = DataType::MAT4_FLOAT32;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT8_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_INT8_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT16_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_INT16_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT32_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_INT32_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_UINT64_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
template <> struct GetMdlShaderType<DataType::MAT4_INT64_NORM> { static constexpr auto Type = DataType::MAT4_INT32_NORM;};
// clang-format on

template <DataType T> struct GetComponentCount;
template <> struct GetComponentCount<DataType::UINT8> { static constexpr auto Value = 1; };
// template <> struct GetComponentCount<DataType::INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::INT64_NORM> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT8> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::VEC2_FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::VEC2_FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::VEC2_UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::VEC2_UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::VEC2_INT64_NORM> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT8> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::VEC3_FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::VEC3_FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::VEC3_UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::VEC3_UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::VEC3_INT64_NORM> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT8> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::VEC4_FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::VEC4_FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::VEC4_UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::VEC4_UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::VEC4_INT64_NORM> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT8> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::MAT2_FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::MAT2_FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::MAT2_UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::MAT2_UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::MAT2_INT64_NORM> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT8> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::MAT3_FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::MAT3_FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::MAT3_UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::MAT3_UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::MAT3_INT64_NORM> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT8> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT8> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT16> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT16> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT32> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT32> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT64> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT64> { using Type = int64_t; };
// template <> struct GetComponentCount<DataType::MAT4_FLOAT32> { using Type = float; };
// template <> struct GetComponentCount<DataType::MAT4_FLOAT64> { using Type = double; };
// template <> struct GetComponentCount<DataType::MAT4_UINT8_NORM> { using Type = uint8_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT8_NORM> { using Type = int8_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT16_NORM> { using Type = uint16_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT16_NORM> { using Type = int16_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT32_NORM> { using Type = uint32_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT32_NORM> { using Type = int32_t; };
// template <> struct GetComponentCount<DataType::MAT4_UINT64_NORM> { using Type = uint64_t; };
// template <> struct GetComponentCount<DataType::MAT4_INT64_NORM> { using Type = int64_t; };

// clang-format off
template <typename RawType, typename TransformedType> struct GetType;
template <> struct GetType<uint8_t, uint8_t> { static constexpr auto Type = DataType::UINT8; };
template <> struct GetType<int8_t, int8_t> { static constexpr auto Type = DataType::INT8; };
template <> struct GetType<uint16_t, uint16_t> { static constexpr auto Type = DataType::UINT16; };
template <> struct GetType<int16_t, int16_t> { static constexpr auto Type = DataType::INT16; };
template <> struct GetType<uint32_t, uint32_t> { static constexpr auto Type = DataType::UINT32; };
template <> struct GetType<int32_t, int32_t> { static constexpr auto Type = DataType::INT32; };
template <> struct GetType<uint64_t, uint64_t> { static constexpr auto Type = DataType::UINT64; };
template <> struct GetType<int64_t, int64_t> { static constexpr auto Type = DataType::INT64; };
template <> struct GetType<float, float> { static constexpr auto Type = DataType::FLOAT32; };
template <> struct GetType<double, double> { static constexpr auto Type = DataType::FLOAT64; };
template <> struct GetType<glm::u8vec2, glm::u8vec2> { static constexpr auto Type = DataType::VEC2_UINT8; };
template <> struct GetType<glm::i8vec2, glm::i8vec2> { static constexpr auto Type = DataType::VEC2_INT8; };
template <> struct GetType<glm::u16vec2, glm::u16vec2> { static constexpr auto Type = DataType::VEC2_UINT16; };
template <> struct GetType<glm::i16vec2, glm::i16vec2> { static constexpr auto Type = DataType::VEC2_INT16; };
template <> struct GetType<glm::u32vec2, glm::u32vec2> { static constexpr auto Type = DataType::VEC2_UINT32; };
template <> struct GetType<glm::i32vec2, glm::i32vec2> { static constexpr auto Type = DataType::VEC2_INT32; };
template <> struct GetType<glm::u64vec2, glm::u64vec2> { static constexpr auto Type = DataType::VEC2_UINT64; };
template <> struct GetType<glm::i64vec2, glm::i64vec2> { static constexpr auto Type = DataType::VEC2_INT64; };
template <> struct GetType<glm::f32vec2, glm::f32vec2> { static constexpr auto Type = DataType::VEC2_FLOAT32; };
template <> struct GetType<glm::f64vec2, glm::f64vec2> { static constexpr auto Type = DataType::VEC2_FLOAT64; };
template <> struct GetType<glm::u8vec3, glm::u8vec3> { static constexpr auto Type = DataType::VEC3_UINT8; };
template <> struct GetType<glm::i8vec3, glm::i8vec3> { static constexpr auto Type = DataType::VEC3_INT8; };
template <> struct GetType<glm::u16vec3, glm::u16vec3> { static constexpr auto Type = DataType::VEC3_UINT16; };
template <> struct GetType<glm::i16vec3, glm::i16vec3> { static constexpr auto Type = DataType::VEC3_INT16; };
template <> struct GetType<glm::u32vec3, glm::u32vec3> { static constexpr auto Type = DataType::VEC3_UINT32; };
template <> struct GetType<glm::i32vec3, glm::i32vec3> { static constexpr auto Type = DataType::VEC3_INT32; };
template <> struct GetType<glm::u64vec3, glm::u64vec3> { static constexpr auto Type = DataType::VEC3_UINT64; };
template <> struct GetType<glm::i64vec3, glm::i64vec3> { static constexpr auto Type = DataType::VEC3_INT64; };
template <> struct GetType<glm::f32vec3, glm::f32vec3> { static constexpr auto Type = DataType::VEC3_FLOAT32; };
template <> struct GetType<glm::f64vec3, glm::f64vec3> { static constexpr auto Type = DataType::VEC3_FLOAT64; };
template <> struct GetType<glm::u8vec4, glm::u8vec4> { static constexpr auto Type = DataType::VEC4_UINT8; };
template <> struct GetType<glm::i8vec4, glm::i8vec4> { static constexpr auto Type = DataType::VEC4_INT8; };
template <> struct GetType<glm::u16vec4, glm::u16vec4> { static constexpr auto Type = DataType::VEC4_UINT16; };
template <> struct GetType<glm::i16vec4, glm::i16vec4> { static constexpr auto Type = DataType::VEC4_INT16; };
template <> struct GetType<glm::u32vec4, glm::u32vec4> { static constexpr auto Type = DataType::VEC4_UINT32; };
template <> struct GetType<glm::i32vec4, glm::i32vec4> { static constexpr auto Type = DataType::VEC4_INT32; };
template <> struct GetType<glm::u64vec4, glm::u64vec4> { static constexpr auto Type = DataType::VEC4_UINT64; };
template <> struct GetType<glm::i64vec4, glm::i64vec4> { static constexpr auto Type = DataType::VEC4_INT64; };
template <> struct GetType<glm::f32vec4, glm::f32vec4> { static constexpr auto Type = DataType::VEC4_FLOAT32; };
template <> struct GetType<glm::f64vec4, glm::f64vec4> { static constexpr auto Type = DataType::VEC4_FLOAT64; };
template <> struct GetType<glm::mat<2, 2, uint8_t>, glm::mat<2, 2, uint8_t>> { static constexpr auto Type = DataType::MAT2_UINT8; };
template <> struct GetType<glm::mat<2, 2, int8_t>, glm::mat<2, 2, int8_t>> { static constexpr auto Type = DataType::MAT2_INT8; };
template <> struct GetType<glm::mat<2, 2, uint16_t>, glm::mat<2, 2, uint16_t>> { static constexpr auto Type = DataType::MAT2_UINT16; };
template <> struct GetType<glm::mat<2, 2, int16_t>, glm::mat<2, 2, int16_t>> { static constexpr auto Type = DataType::MAT2_INT16; };
template <> struct GetType<glm::mat<2, 2, uint32_t>, glm::mat<2, 2, uint32_t>> { static constexpr auto Type = DataType::MAT2_UINT32; };
template <> struct GetType<glm::mat<2, 2, int32_t>, glm::mat<2, 2, int32_t>> { static constexpr auto Type = DataType::MAT2_INT32; };
template <> struct GetType<glm::mat<2, 2, uint64_t>, glm::mat<2, 2, uint64_t>> { static constexpr auto Type = DataType::MAT2_UINT64; };
template <> struct GetType<glm::mat<2, 2, int64_t>, glm::mat<2, 2, int64_t>> { static constexpr auto Type = DataType::MAT2_INT64; };
template <> struct GetType<glm::f32mat2, glm::f32mat2> { static constexpr auto Type = DataType::MAT2_FLOAT32; };
template <> struct GetType<glm::f64mat2, glm::f32mat2> { static constexpr auto Type = DataType::MAT2_FLOAT64; };
template <> struct GetType<glm::mat<3, 3, uint8_t>, glm::mat<3, 3, uint8_t>> { static constexpr auto Type = DataType::MAT3_UINT8; };
template <> struct GetType<glm::mat<3, 3, int8_t>, glm::mat<3, 3, int8_t>> { static constexpr auto Type = DataType::MAT3_INT8; };
template <> struct GetType<glm::mat<3, 3, uint16_t>, glm::mat<3, 3, uint16_t>> { static constexpr auto Type = DataType::MAT3_UINT16; };
template <> struct GetType<glm::mat<3, 3, int16_t>, glm::mat<3, 3, int16_t>> { static constexpr auto Type = DataType::MAT3_INT16; };
template <> struct GetType<glm::mat<3, 3, uint32_t>, glm::mat<3, 3, uint32_t>> { static constexpr auto Type = DataType::MAT3_UINT32; };
template <> struct GetType<glm::mat<3, 3, int32_t>, glm::mat<3, 3, int32_t>> { static constexpr auto Type = DataType::MAT3_INT32; };
template <> struct GetType<glm::mat<3, 3, uint64_t>, glm::mat<3, 3, uint64_t>> { static constexpr auto Type = DataType::MAT3_UINT64; };
template <> struct GetType<glm::mat<3, 3, int64_t>, glm::mat<3, 3, int64_t>> { static constexpr auto Type = DataType::MAT3_INT64; };
template <> struct GetType<glm::f32mat3, glm::f32mat3> { static constexpr auto Type = DataType::MAT3_FLOAT32; };
template <> struct GetType<glm::f64mat3, glm::f64mat3> { static constexpr auto Type = DataType::MAT3_FLOAT64; };
template <> struct GetType<glm::mat<4, 4, uint8_t>, glm::mat<4, 4, uint8_t>> { static constexpr auto Type = DataType::MAT4_UINT8; };
template <> struct GetType<glm::mat<4, 4, int8_t>, glm::mat<4, 4, int8_t>> { static constexpr auto Type = DataType::MAT4_INT8; };
template <> struct GetType<glm::mat<4, 4, uint16_t>, glm::mat<4, 4, uint16_t>> { static constexpr auto Type = DataType::MAT4_UINT16; };
template <> struct GetType<glm::mat<4, 4, int16_t>, glm::mat<4, 4, int16_t>> { static constexpr auto Type = DataType::MAT4_INT16; };
template <> struct GetType<glm::mat<4, 4, uint32_t>, glm::mat<4, 4, uint32_t>> { static constexpr auto Type = DataType::MAT4_UINT32; };
template <> struct GetType<glm::mat<4, 4, int32_t>, glm::mat<4, 4, int32_t>> { static constexpr auto Type = DataType::MAT4_INT32; };
template <> struct GetType<glm::mat<4, 4, uint64_t>, glm::mat<4, 4, uint64_t>> { static constexpr auto Type = DataType::MAT4_UINT64; };
template <> struct GetType<glm::mat<4, 4, int64_t>, glm::mat<4, 4, int64_t>> { static constexpr auto Type = DataType::MAT4_INT64; };
template <> struct GetType<glm::f32mat4, glm::f32mat4> { static constexpr auto Type = DataType::MAT4_FLOAT32; };
template <> struct GetType<glm::f64mat4, glm::f64mat4> { static constexpr auto Type = DataType::MAT4_FLOAT64; };
template <> struct GetType<uint8_t, double> { static constexpr auto Type = DataType::UINT8_NORM; };
template <> struct GetType<int8_t, double> { static constexpr auto Type = DataType::INT8_NORM; };
template <> struct GetType<uint16_t, double> { static constexpr auto Type = DataType::UINT16_NORM; };
template <> struct GetType<int16_t, double> { static constexpr auto Type = DataType::INT16_NORM; };
template <> struct GetType<uint32_t, double> { static constexpr auto Type = DataType::UINT32_NORM; };
template <> struct GetType<int32_t, double> { static constexpr auto Type = DataType::INT32_NORM; };
template <> struct GetType<uint64_t, double> { static constexpr auto Type = DataType::UINT64_NORM; };
template <> struct GetType<int64_t, double> { static constexpr auto Type = DataType::INT64_NORM; };
template <> struct GetType<glm::u8vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT8_NORM; };
template <> struct GetType<glm::i8vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT8_NORM; };
template <> struct GetType<glm::u16vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT16_NORM; };
template <> struct GetType<glm::i16vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT16_NORM; };
template <> struct GetType<glm::u32vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT32_NORM; };
template <> struct GetType<glm::i32vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT32_NORM; };
template <> struct GetType<glm::u64vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_UINT64_NORM; };
template <> struct GetType<glm::i64vec2, glm::dvec2> { static constexpr auto Type = DataType::VEC2_INT64_NORM; };
template <> struct GetType<glm::u8vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT8_NORM; };
template <> struct GetType<glm::i8vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT8_NORM; };
template <> struct GetType<glm::u16vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT16_NORM; };
template <> struct GetType<glm::i16vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT16_NORM; };
template <> struct GetType<glm::u32vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT32_NORM; };
template <> struct GetType<glm::i32vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT32_NORM; };
template <> struct GetType<glm::u64vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_UINT64_NORM; };
template <> struct GetType<glm::i64vec3, glm::dvec3> { static constexpr auto Type = DataType::VEC3_INT64_NORM; };
template <> struct GetType<glm::u8vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT8_NORM; };
template <> struct GetType<glm::i8vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT8_NORM; };
template <> struct GetType<glm::u16vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT16_NORM; };
template <> struct GetType<glm::i16vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT16_NORM; };
template <> struct GetType<glm::u32vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT32_NORM; };
template <> struct GetType<glm::i32vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT32_NORM; };
template <> struct GetType<glm::u64vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_UINT64_NORM; };
template <> struct GetType<glm::i64vec4, glm::dvec4> { static constexpr auto Type = DataType::VEC4_INT64_NORM; };
template <> struct GetType<glm::mat<2, 2, uint8_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT8_NORM; };
template <> struct GetType<glm::mat<2, 2, int8_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT8_NORM; };
template <> struct GetType<glm::mat<2, 2, uint16_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT16_NORM; };
template <> struct GetType<glm::mat<2, 2, int16_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT16_NORM; };
template <> struct GetType<glm::mat<2, 2, uint32_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT32_NORM; };
template <> struct GetType<glm::mat<2, 2, int32_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT32_NORM; };
template <> struct GetType<glm::mat<2, 2, uint64_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_UINT64_NORM; };
template <> struct GetType<glm::mat<2, 2, int64_t>, glm::dmat2> { static constexpr auto Type = DataType::MAT2_INT64_NORM; };
template <> struct GetType<glm::mat<3, 3, uint8_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT8_NORM; };
template <> struct GetType<glm::mat<3, 3, int8_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT8_NORM; };
template <> struct GetType<glm::mat<3, 3, uint16_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT16_NORM; };
template <> struct GetType<glm::mat<3, 3, int16_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT16_NORM; };
template <> struct GetType<glm::mat<3, 3, uint32_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT32_NORM; };
template <> struct GetType<glm::mat<3, 3, int32_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT32_NORM; };
template <> struct GetType<glm::mat<3, 3, uint64_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_UINT64_NORM; };
template <> struct GetType<glm::mat<3, 3, int64_t>, glm::dmat3> { static constexpr auto Type = DataType::MAT3_INT64_NORM; };
template <> struct GetType<glm::mat<4, 4, uint8_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT8_NORM; };
template <> struct GetType<glm::mat<4, 4, int8_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT8_NORM; };
template <> struct GetType<glm::mat<4, 4, uint16_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT16_NORM; };
template <> struct GetType<glm::mat<4, 4, int16_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT16_NORM; };
template <> struct GetType<glm::mat<4, 4, uint32_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT32_NORM; };
template <> struct GetType<glm::mat<4, 4, int32_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT32_NORM; };
template <> struct GetType<glm::mat<4, 4, uint64_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_UINT64_NORM; };
template <> struct GetType<glm::mat<4, 4, int64_t>, glm::dmat4> { static constexpr auto Type = DataType::MAT4_INT64_NORM; };
// clang-format on

// Only specialized for valid glTF vertex attribute types and valid fabric primvar types.
// Excludes 32 and 64-bit integer types, 64-bit floating point types, and matrix types.
// If we want to add support for matrix types in the future we can pack each row in a separate primvar and reassemble in MDL.
template <DataType T> struct GetPrimvarTypeImpl;

// Integer primvar lookup in MDL doesn't seem to work so cast all data types to float. This is safe to do since
// FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
// overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways. There is some overhead for UINT8 values
// which could be stored as eUChar.
template <> struct GetPrimvarTypeImpl<DataType::UINT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::FLOAT32> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::UINT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::INT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_UINT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC2_INT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_UINT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC3_INT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_UINT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<DataType::VEC4_INT16_NORM> { using Type = glm::f32vec4; };

template <DataType T> using GetPrimvarType = typename GetPrimvarTypeImpl<T>::Type;

constexpr omni::fabric::BaseDataType getPrimvarBaseDataType(DataType type) {
    // assert(isPrimvarType(type));

    // const auto componentType = getComponentType(getPrimvarType(type));
    // switch (componentType) {
    //     case DataType::UINT8:
    //         return omni::fabric::BaseDataType::eUChar;
    //     case DataType::INT8:
    //     case DataType::UINT16:
    //     case DataType::INT16:
    //         return omni::fabric::BaseDataType::eInt;
    //     case DataType::FLOAT32:
    //         return omni::fabric::BaseDataType::eFloat;
    //     default:
    //         // Not a valid vertex attribute type
    //         assert(false);
    //         return omni::fabric::BaseDataType::eUnknown;
    // }

    (void)type;
    return {};
}

constexpr omni::fabric::Type getFabricPrimvarType(DataType type) {
    // assert(isPrimvarType(type));
    // const auto baseDataType = getPrimvarBaseDataType(type);
    // const auto componentCount = getComponentCount(type);
    // return {baseDataType, static_cast<uint8_t>(componentCount), 1, omni::fabric::AttributeRole::eNone};

    (void)type;
    return {};
}

} // namespace cesium::omniverse
