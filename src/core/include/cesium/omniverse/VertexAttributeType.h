#pragma once

#include <glm/glm.hpp>
#include <omni/fabric/FabricTypes.h>

#include <optional>
#include <string>

namespace cesium::omniverse {

enum class VertexAttributeType {
    UINT8,
    INT8,
    UINT16,
    INT16,
    FLOAT32,
    VEC2_UINT8,
    VEC2_INT8,
    VEC2_UINT16,
    VEC2_INT16,
    VEC2_FLOAT32,
    VEC3_UINT8,
    VEC3_INT8,
    VEC3_UINT16,
    VEC3_INT16,
    VEC3_FLOAT32,
    VEC4_UINT8,
    VEC4_INT8,
    VEC4_UINT16,
    VEC4_INT16,
    VEC4_FLOAT32,
};

std::optional<VertexAttributeType> getVertexAttributeTypeFromGltf(const std::string& type, int32_t componentType);
omni::fabric::Type getFabricType(VertexAttributeType type);

template <VertexAttributeType T> struct GetNativeTypeImpl;
template <> struct GetNativeTypeImpl<VertexAttributeType::UINT8> { using Type = uint8_t; };
template <> struct GetNativeTypeImpl<VertexAttributeType::INT8> { using Type = int8_t; };
template <> struct GetNativeTypeImpl<VertexAttributeType::UINT16> { using Type = uint16_t; };
template <> struct GetNativeTypeImpl<VertexAttributeType::INT16> { using Type = int16_t; };
template <> struct GetNativeTypeImpl<VertexAttributeType::FLOAT32> { using Type = float; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC2_UINT8> { using Type = glm::u8vec2; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC2_INT8> { using Type = glm::i8vec2; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC2_UINT16> { using Type = glm::u16vec2; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC2_INT16> { using Type = glm::i16vec2; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC3_UINT8> { using Type = glm::u8vec3; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC3_INT8> { using Type = glm::i8vec3; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC3_UINT16> { using Type = glm::u16vec3; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC3_INT16> { using Type = glm::i16vec3; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC4_UINT8> { using Type = glm::u8vec4; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC4_INT8> { using Type = glm::i8vec4; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC4_UINT16> { using Type = glm::u16vec4; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC4_INT16> { using Type = glm::i16vec4; };
template <> struct GetNativeTypeImpl<VertexAttributeType::VEC4_FLOAT32> { using Type = glm::f32vec4; };

// Integer primvar lookup doesn't seem to work so cast all data types to float. This is safe to do since
// FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
// overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways.
template <VertexAttributeType T> struct GetFabricTypeImpl;
template <> struct GetFabricTypeImpl<VertexAttributeType::UINT8> { using Type = float; };
template <> struct GetFabricTypeImpl<VertexAttributeType::INT8> { using Type = float; };
template <> struct GetFabricTypeImpl<VertexAttributeType::UINT16> { using Type = float; };
template <> struct GetFabricTypeImpl<VertexAttributeType::INT16> { using Type = float; };
template <> struct GetFabricTypeImpl<VertexAttributeType::FLOAT32> { using Type = float; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC2_UINT8> { using Type = glm::f32vec2; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC2_INT8> { using Type = glm::f32vec2; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC2_UINT16> { using Type = glm::f32vec2; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC2_INT16> { using Type = glm::f32vec2; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC2_FLOAT32> { using Type = glm::f32vec2; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC3_UINT8> { using Type = glm::f32vec3; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC3_INT8> { using Type = glm::f32vec3; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC3_UINT16> { using Type = glm::f32vec3; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC3_INT16> { using Type = glm::f32vec3; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC3_FLOAT32> { using Type = glm::f32vec3; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC4_UINT8> { using Type = glm::f32vec4; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC4_INT8> { using Type = glm::f32vec4; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC4_UINT16> { using Type = glm::f32vec4; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC4_INT16> { using Type = glm::f32vec4; };
template <> struct GetFabricTypeImpl<VertexAttributeType::VEC4_FLOAT32> { using Type = glm::f32vec4; };

template <VertexAttributeType T> using GetNativeType = typename GetNativeTypeImpl<T>::Type;
template <VertexAttributeType T> using GetFabricType = typename GetFabricTypeImpl<T>::Type;

} // namespace cesium::omniverse
