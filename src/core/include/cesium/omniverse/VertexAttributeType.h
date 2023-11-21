#pragma once

#include <glm/glm.hpp>
#include <omni/fabric/Type.h>

namespace cesium::omniverse {

enum class VertexAttributeType {
    UINT8,
    INT8,
    UINT16,
    INT16,
    FLOAT32,
    UINT8_NORM,
    INT8_NORM,
    UINT16_NORM,
    INT16_NORM,
    VEC2_UINT8,
    VEC2_INT8,
    VEC2_UINT16,
    VEC2_INT16,
    VEC2_FLOAT32,
    VEC2_UINT8_NORM,
    VEC2_INT8_NORM,
    VEC2_UINT16_NORM,
    VEC2_INT16_NORM,
    VEC3_UINT8,
    VEC3_INT8,
    VEC3_UINT16,
    VEC3_INT16,
    VEC3_FLOAT32,
    VEC3_UINT8_NORM,
    VEC3_INT8_NORM,
    VEC3_UINT16_NORM,
    VEC3_INT16_NORM,
    VEC4_UINT8,
    VEC4_INT8,
    VEC4_UINT16,
    VEC4_INT16,
    VEC4_FLOAT32,
    VEC4_UINT8_NORM,
    VEC4_INT8_NORM,
    VEC4_UINT16_NORM,
    VEC4_INT16_NORM
};

constexpr auto VertexAttributeTypeCount = static_cast<uint64_t>(VertexAttributeType::VEC4_INT16_NORM) + 1;

template <VertexAttributeType T> struct GetRawTypeImpl;
template <> struct GetRawTypeImpl<VertexAttributeType::UINT8> { using Type = uint8_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::INT8> { using Type = int8_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::UINT16> { using Type = uint16_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::INT16> { using Type = int16_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::FLOAT32> { using Type = float; };
template <> struct GetRawTypeImpl<VertexAttributeType::UINT8_NORM> { using Type = uint8_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::INT8_NORM> { using Type = int8_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::UINT16_NORM> { using Type = uint16_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::INT16_NORM> { using Type = int16_t; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_UINT8> { using Type = glm::u8vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_INT8> { using Type = glm::i8vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_UINT16> { using Type = glm::u16vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_INT16> { using Type = glm::i16vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_UINT8_NORM> { using Type = glm::u8vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_INT8_NORM> { using Type = glm::i8vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_UINT16_NORM> { using Type = glm::u16vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC2_INT16_NORM> { using Type = glm::i16vec2; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_UINT8> { using Type = glm::u8vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_INT8> { using Type = glm::i8vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_UINT16> { using Type = glm::u16vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_INT16> { using Type = glm::i16vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_UINT8_NORM> { using Type = glm::u8vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_INT8_NORM> { using Type = glm::i8vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_UINT16_NORM> { using Type = glm::u16vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC3_INT16_NORM> { using Type = glm::i16vec3; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_UINT8> { using Type = glm::u8vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_INT8> { using Type = glm::i8vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_UINT16> { using Type = glm::u16vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_INT16> { using Type = glm::i16vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_UINT8_NORM> { using Type = glm::u8vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_INT8_NORM> { using Type = glm::i8vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_UINT16_NORM> { using Type = glm::u16vec4; };
template <> struct GetRawTypeImpl<VertexAttributeType::VEC4_INT16_NORM> { using Type = glm::i16vec4; };
template <VertexAttributeType T> using GetRawType = typename GetRawTypeImpl<T>::Type;

// clang-format off
template <VertexAttributeType T> struct GetComponentCount;
template <> struct GetComponentCount<VertexAttributeType::UINT8> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::INT8> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::UINT16> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::INT16> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::FLOAT32> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::UINT8_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::INT8_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::UINT16_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::INT16_NORM> { static constexpr auto ComponentCount = 1; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_UINT8> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_INT8> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_UINT16> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_INT16> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_FLOAT32> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_UINT8_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_INT8_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_UINT16_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC2_INT16_NORM> { static constexpr auto ComponentCount = 2; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_UINT8> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_INT8> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_UINT16> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_INT16> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_FLOAT32> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_UINT8_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_INT8_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_UINT16_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC3_INT16_NORM> { static constexpr auto ComponentCount = 3; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_UINT8> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_INT8> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_UINT16> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_INT16> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_FLOAT32> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_UINT8_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_INT8_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_UINT16_NORM> { static constexpr auto ComponentCount = 4; };
template <> struct GetComponentCount<VertexAttributeType::VEC4_INT16_NORM> { static constexpr auto ComponentCount = 4; };
// clang-format on

// Integer primvar lookup in MDL doesn't seem to work so cast all data types to float. This is safe to do since
// FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
// overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways. There is some overhead for UINT8 values
// which could be stored as eUChar.
template <VertexAttributeType T> struct GetPrimvarTypeImpl;
template <> struct GetPrimvarTypeImpl<VertexAttributeType::UINT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::INT8> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::UINT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::INT16> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::FLOAT32> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::UINT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::INT8_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::UINT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::INT16_NORM> { using Type = float; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_UINT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_INT8> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_UINT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_INT16> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_UINT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_INT8_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_UINT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC2_INT16_NORM> { using Type = glm::f32vec2; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_UINT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_INT8> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_UINT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_INT16> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_UINT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_INT8_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_UINT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC3_INT16_NORM> { using Type = glm::f32vec3; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_UINT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_INT8> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_UINT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_INT16> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_FLOAT32> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_UINT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_INT8_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_UINT16_NORM> { using Type = glm::f32vec4; };
template <> struct GetPrimvarTypeImpl<VertexAttributeType::VEC4_INT16_NORM> { using Type = glm::f32vec4; };
template <VertexAttributeType T> using GetPrimvarType = typename GetPrimvarTypeImpl<T>::Type;

// clang-format off
template <VertexAttributeType T> struct GetPrimvarBaseDataType;
template <> struct GetPrimvarBaseDataType<VertexAttributeType::UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC2_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC3_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_UINT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_INT8> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_UINT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_INT16> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_FLOAT32> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_UINT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_INT8_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_UINT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
template <> struct GetPrimvarBaseDataType<VertexAttributeType::VEC4_INT16_NORM> { static constexpr auto BaseDataType = omni::fabric::BaseDataType::eFloat; };
// clang-format on

template <typename L, std::size_t... I> const auto& dispatchImpl(std::index_sequence<I...>, L lambda) {
    static decltype(lambda(std::integral_constant<VertexAttributeType, VertexAttributeType::UINT8>{})) array[] = {
        lambda(std::integral_constant<VertexAttributeType, VertexAttributeType(I)>{})...};
    return array;
}
template <typename L, typename... P> auto dispatch(L lambda, VertexAttributeType n, P&&... p) {
    const auto& array = dispatchImpl(std::make_index_sequence<VertexAttributeTypeCount>{}, lambda);
    return array[static_cast<size_t>(n)](std::forward<P>(p)...);
}

// This allows us to call an enum templated function based on a runtime enum value
#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE(FUNCTION_NAME, TYPE, ...) \
    dispatch([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE, __VA_ARGS__)

// In C++ 20 we don't need this second define
#define CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE_NO_ARGS(FUNCTION_NAME, TYPE) \
    dispatch([](auto i) { return FUNCTION_NAME<i.value>; }, TYPE)

} // namespace cesium::omniverse
