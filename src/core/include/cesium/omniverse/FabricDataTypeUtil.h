#pragma once

#include "cesium/omniverse/CppUtil.h"
#include "cesium/omniverse/DataType.h"

#include <CesiumGltf/PropertyTypeTraits.h>
#include <carb/RenderingTypes.h>
#include <glm/glm.hpp>
#include <omni/fabric/Type.h>

namespace cesium::omniverse::detail {

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

} // namespace cesium::omniverse::detail

namespace cesium::omniverse::FabricDataTypeUtil {

template <DataType T> constexpr omni::fabric::BaseDataType getPrimvarBaseDataType() {
    return detail::GetPrimvarBaseDataTypeImpl<T>::BaseDataType;
};

inline omni::fabric::BaseDataType getPrimvarBaseDataType(DataType type) {
    return CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(getPrimvarBaseDataType, type);
}

} // namespace cesium::omniverse::FabricDataTypeUtil
