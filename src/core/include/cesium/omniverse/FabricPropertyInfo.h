#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/DataTypeUtil.h"
#include "cesium/omniverse/FabricTextureInfo.h"

namespace cesium::omniverse {

template <DataType T> struct FabricPropertyInfo {
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> offset;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> scale;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> min;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> max;
    bool required;
    std::optional<DataTypeUtil::GetNativeType<T>> noData;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> defaultValue;
};

template <DataType T> struct FabricPropertyAttributePropertyInfo {
    static constexpr auto Type = T;
    std::string attribute;
    FabricPropertyInfo<T> propertyInfo;
};

template <DataType T> struct FabricPropertyTexturePropertyInfo {
    static constexpr auto Type = T;
    FabricTextureInfo textureInfo;
    uint64_t textureIndex;
    FabricPropertyInfo<T> propertyInfo;
};

template <DataType T> struct FabricPropertyTablePropertyInfo {
    static constexpr auto Type = T;
    uint64_t featureIdSetIndex;
    FabricPropertyInfo<T> propertyInfo;
};

} // namespace cesium::omniverse
