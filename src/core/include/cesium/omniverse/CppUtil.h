#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace cesium::omniverse::CppUtil {

template <typename T, typename L, std::size_t... I> const auto& dispatchImpl(std::index_sequence<I...>, L lambda) {
    static decltype(lambda(std::integral_constant<T, T(0)>{})) array[] = {lambda(std::integral_constant<T, T(I)>{})...};
    return array;
}
template <uint64_t T_COUNT, typename T, typename L, typename... P> auto dispatch(L lambda, T n, P&&... p) {
    const auto& array = dispatchImpl<T>(std::make_index_sequence<T_COUNT>{}, lambda);
    return array[static_cast<size_t>(n)](std::forward<P>(p)...);
}

template <typename T> const T& defaultValue(const T* value, const T& defaultValue) {
    return value == nullptr ? defaultValue : *value;
}

template <typename T, typename U> T defaultValue(const std::optional<U>& optional, const T& defaultValue) {
    return optional.has_value() ? static_cast<T>(optional.value()) : defaultValue;
}

template <typename T> size_t getIndexFromRef(const std::vector<T>& vector, const T& item) {
    return static_cast<size_t>(&item - vector.data());
};

template <typename T, typename U> std::optional<T> castOptional(const std::optional<U>& optional) {
    return optional.has_value() ? std::make_optional(static_cast<T>(optional.value())) : std::nullopt;
}

template <typename T> uint64_t indexOf(const std::vector<T>& vector, const T& value) {
    return static_cast<uint64_t>(std::distance(vector.begin(), std::find(vector.begin(), vector.end(), value)));
}

} // namespace cesium::omniverse::CppUtil
