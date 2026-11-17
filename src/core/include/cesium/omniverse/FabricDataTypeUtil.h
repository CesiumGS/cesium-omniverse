#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/DataTypeUtil.h"

#include <CesiumGltf/PropertyTypeTraits.h>
#include <carb/RenderingTypes.h>
#include <glm/glm.hpp>
#include <omni/fabric/Type.h>

namespace cesium::omniverse::detail {

// clang-format off
template <DataType T> struct GetFabricBaseDataTypeImpl;
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC2_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC3_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::VEC4_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT2_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT3_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_FLOAT64> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eDouble; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUChar; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT32_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eUInt64; };
template <> struct GetFabricBaseDataTypeImpl<DataType::MAT4_INT64_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eInt64; };

// Downcast 64 bit to 32 bit
// eRGB8_UINT, eRGB8_SINT, eRGB16_UINT, eRGB16_SINT do not exist, so use the RGBA equivalent
template <DataType T> struct GetCarbFormatImpl;
template <> struct GetCarbFormatImpl<DataType::UINT8> { static constexpr auto CarbFormat = carb::Format::eR8_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT8> { static constexpr auto CarbFormat = carb::Format::eR8_SINT; };
template <> struct GetCarbFormatImpl<DataType::UINT16> { static constexpr auto CarbFormat = carb::Format::eR16_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT16> { static constexpr auto CarbFormat = carb::Format::eR16_SINT; };
template <> struct GetCarbFormatImpl<DataType::UINT32> { static constexpr auto CarbFormat = carb::Format::eR32_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT32> { static constexpr auto CarbFormat = carb::Format::eR32_SINT; };
template <> struct GetCarbFormatImpl<DataType::UINT64> { static constexpr auto CarbFormat = carb::Format::eR32_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT64> { static constexpr auto CarbFormat = carb::Format::eR32_SINT; };
template <> struct GetCarbFormatImpl<DataType::FLOAT32> { static constexpr auto CarbFormat = carb::Format::eR32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::FLOAT64> { static constexpr auto CarbFormat = carb::Format::eR32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eR8_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eR8_SINT; };
template <> struct GetCarbFormatImpl<DataType::UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eR16_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eR16_SINT; };
template <> struct GetCarbFormatImpl<DataType::UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eR32_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eR32_SINT; };
template <> struct GetCarbFormatImpl<DataType::UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eR32_UINT; };
template <> struct GetCarbFormatImpl<DataType::INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eR32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT8> { static constexpr auto CarbFormat = carb::Format::eRG8_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT8> { static constexpr auto CarbFormat = carb::Format::eRG8_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT16> { static constexpr auto CarbFormat = carb::Format::eRG16_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT16> { static constexpr auto CarbFormat = carb::Format::eRG16_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT32> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT32> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT64> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT64> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_FLOAT32> { static constexpr auto CarbFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_FLOAT64> { static constexpr auto CarbFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRG8_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRG8_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRG16_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRG16_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC2_INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT32> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT32> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT64> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT64> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_FLOAT32> { static constexpr auto CarbFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_FLOAT64> { static constexpr auto CarbFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC3_INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT32> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT32> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT64> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT64> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_FLOAT32> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_FLOAT64> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::VEC4_INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT8> { static constexpr auto CarbFormat = carb::Format::eRG8_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT8> { static constexpr auto CarbFormat = carb::Format::eRG8_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT16> { static constexpr auto CarbFormat = carb::Format::eRG16_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT16> { static constexpr auto CarbFormat = carb::Format::eRG16_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT32> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT32> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT64> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT64> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_FLOAT32> { static constexpr auto CarbFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_FLOAT64> { static constexpr auto CarbFormat = carb::Format::eRG32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRG8_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRG8_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRG16_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRG16_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT2_INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRG32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT32> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT32> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT64> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT64> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_FLOAT32> { static constexpr auto CarbFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_FLOAT64> { static constexpr auto CarbFormat = carb::Format::eRGB32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT3_INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGB32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT8> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT16> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT32> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT32> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT64> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT64> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_FLOAT32> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_FLOAT64> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SFLOAT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT8_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA8_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT16_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA16_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT32_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_UINT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_UINT; };
template <> struct GetCarbFormatImpl<DataType::MAT4_INT64_NORM> { static constexpr auto CarbFormat = carb::Format::eRGBA32_SINT; };
// clang-format on

} // namespace cesium::omniverse::detail

namespace cesium::omniverse::FabricDataTypeUtil {

template <DataType T> constexpr omni::fabric::BaseDataType getFabricBaseDataType() {
    return detail::GetFabricBaseDataTypeImpl<T>::BaseDataType;
};

inline omni::fabric::BaseDataType getFabricBaseDataType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getFabricBaseDataType, type);
}

template <DataType T> constexpr carb::Format getCarbFormat() {
    return detail::GetCarbFormatImpl<T>::CarbFormat;
};

inline carb::Format getCarbFormat(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getCarbFormat, type);
};

} // namespace cesium::omniverse::FabricDataTypeUtil
