#pragma once

#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/DataTypeUtil.h"
#include "cesium/omniverse/TextureInfo.h"

namespace cesium::omniverse {

template <DataType T> struct PropertyInfo {
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> offset;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> scale;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> min;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> max;
    bool required;
    std::optional<DataTypeUtil::GetNativeType<T>> noData;
    std::optional<DataTypeUtil::GetNativeType<DataTypeUtil::getTransformedType<T>()>> defaultValue;
};

template <DataType T> struct PropertyAttributePropertyInfo {
    static constexpr auto Type = T;
    std::string attribute;
    PropertyInfo<T> propertyInfo;
};

template <DataType T> struct PropertyTexturePropertyInfo {
    static constexpr auto Type = T;
    TextureInfo textureInfo;
    uint64_t textureIndex;
    PropertyInfo<T> propertyInfo;
};

template <DataType T> struct PropertyTablePropertyInfo {
    static constexpr auto Type = T;
    uint64_t featureIdSetIndex;
    PropertyInfo<T> propertyInfo;
};

} // namespace cesium::omniverse
