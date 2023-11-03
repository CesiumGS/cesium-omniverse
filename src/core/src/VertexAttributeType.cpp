#include "cesium/omniverse/VertexAttributeType.h"

#include <CesiumGltf/Accessor.h>

namespace cesium::omniverse {

namespace {

uint64_t getComponentCount(VertexAttributeType type) {
    switch (type) {
        case VertexAttributeType::UINT8:
        case VertexAttributeType::INT8:
        case VertexAttributeType::UINT16:
        case VertexAttributeType::INT16:
        case VertexAttributeType::FLOAT32:
            return 1;
        case VertexAttributeType::VEC2_UINT8:
        case VertexAttributeType::VEC2_INT8:
        case VertexAttributeType::VEC2_UINT16:
        case VertexAttributeType::VEC2_INT16:
        case VertexAttributeType::VEC2_FLOAT32:
            return 2;
        case VertexAttributeType::VEC3_UINT8:
        case VertexAttributeType::VEC3_INT8:
        case VertexAttributeType::VEC3_UINT16:
        case VertexAttributeType::VEC3_INT16:
        case VertexAttributeType::VEC3_FLOAT32:
            return 3;
        case VertexAttributeType::VEC4_UINT8:
        case VertexAttributeType::VEC4_INT8:
        case VertexAttributeType::VEC4_UINT16:
        case VertexAttributeType::VEC4_INT16:
        case VertexAttributeType::VEC4_FLOAT32:
            return 4;
    }

    // Unreachable code. All enum cases are handled above.
    assert(false);
    return 0;
}

omni::fabric::BaseDataType getFabricBaseDataType() {
    // Integer primvar lookup doesn't seem to work so cast all data types to float. This is safe to do since
    // FLOAT32 can represent all possible UINT8, INT8, UINT16, and INT16 values. Also not a significant memory
    // overhead since Fabric doesn't support INT8, UINT16, and INT16 types anyways.
    return omni::fabric::BaseDataType::eFloat;
}

} // namespace

std::optional<VertexAttributeType> getVertexAttributeTypeFromGltf(const std::string& type, int32_t componentType) {
    if (type == CesiumGltf::Accessor::Type::SCALAR) {
        if (componentType == CesiumGltf::Accessor::ComponentType::BYTE) {
            return VertexAttributeType::INT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
            return VertexAttributeType::UINT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::SHORT) {
            return VertexAttributeType::INT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
            return VertexAttributeType::UINT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::FLOAT) {
            return VertexAttributeType::FLOAT32;
        }
    } else if (type == CesiumGltf::Accessor::Type::VEC2) {
        if (componentType == CesiumGltf::Accessor::ComponentType::BYTE) {
            return VertexAttributeType::VEC2_INT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
            return VertexAttributeType::VEC2_UINT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::SHORT) {
            return VertexAttributeType::VEC2_INT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
            return VertexAttributeType::VEC2_UINT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::FLOAT) {
            return VertexAttributeType::VEC2_FLOAT32;
        }
    } else if (type == CesiumGltf::Accessor::Type::VEC3) {
        if (componentType == CesiumGltf::Accessor::ComponentType::BYTE) {
            return VertexAttributeType::VEC3_INT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
            return VertexAttributeType::VEC3_UINT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::SHORT) {
            return VertexAttributeType::VEC3_INT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
            return VertexAttributeType::VEC3_UINT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::FLOAT) {
            return VertexAttributeType::VEC3_FLOAT32;
        }
    } else if (type == CesiumGltf::Accessor::Type::VEC4) {
        if (componentType == CesiumGltf::Accessor::ComponentType::BYTE) {
            return VertexAttributeType::VEC4_INT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE) {
            return VertexAttributeType::VEC4_UINT8;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::SHORT) {
            return VertexAttributeType::VEC4_INT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT) {
            return VertexAttributeType::VEC4_UINT16;
        } else if (componentType == CesiumGltf::Accessor::ComponentType::FLOAT) {
            return VertexAttributeType::VEC4_FLOAT32;
        }
    }

    // Cases where nullopt is returned
    // - componentType is UNSIGNED_INT. UNSIGNED_INT is not an allowed glTF vertex attribute componentType.
    // - type is MAT2, MAT3, MAT4. Matrix types are not supported primvar types.
    return std::nullopt;
}

omni::fabric::Type getFabricType(VertexAttributeType type) {
    const auto baseDataType = getFabricBaseDataType();
    const auto componentCount = getComponentCount(type);
    return {baseDataType, static_cast<uint8_t>(componentCount), 1, omni::fabric::AttributeRole::eNone};
}

} // namespace cesium::omniverse
