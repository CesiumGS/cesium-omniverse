#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

namespace cesium::omniverse::CppUtil {

template <typename T, typename L, uint64_t... I> const auto& dispatchImpl(std::index_sequence<I...>, L lambda) {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    static decltype(lambda(std::integral_constant<T, T(0)>{})) array[] = {lambda(std::integral_constant<T, T(I)>{})...};
    return array;
}

template <uint64_t T_COUNT, typename T, typename L, typename... P> auto dispatch(L lambda, T n, P&&... p) {
    const auto& array = dispatchImpl<T>(std::make_index_sequence<T_COUNT>{}, lambda);
    return array[static_cast<uint64_t>(n)](std::forward<P>(p)...);
}

template <typename T> const T& defaultValue(const T* pValue, const T& defaultValue) {
    return pValue ? *pValue : defaultValue;
}

template <typename T, typename U> T defaultValue(const std::optional<U>& optional, const T& defaultValue) {
    return optional.has_value() ? static_cast<T>(optional.value()) : defaultValue;
}

template <typename T> uint64_t getIndexFromRef(const std::vector<T>& vector, const T& item) {
    return static_cast<uint64_t>(&item - vector.data());
};

template <typename T, typename U> std::optional<T> castOptional(const std::optional<U>& optional) {
    return optional.has_value() ? std::make_optional(static_cast<T>(optional.value())) : std::nullopt;
}

template <typename T, typename U> uint64_t indexOf(const std::vector<T>& vector, const U& value) {
    return static_cast<uint64_t>(std::distance(vector.begin(), std::find(vector.begin(), vector.end(), value)));
}

template <typename T, typename U> uint64_t indexOfByMember(const std::vector<T>& vector, U T::*member, const U& value) {
    return static_cast<uint64_t>(
        std::distance(vector.begin(), std::find_if(vector.begin(), vector.end(), [&value, &member](const auto& item) {
                          return item.*member == value;
                      })));
}

template <typename T, typename U> bool contains(const std::vector<T>& vector, const U& value) {
    return std::find(vector.begin(), vector.end(), value) != vector.end();
}

// In C++ 20 we can use std::ranges::common_range instead of having a separate version for std::array
template <typename T, size_t C, typename U> bool contains(const std::array<T, C>& array, const U& value) {
    return std::find(array.begin(), array.end(), value) != array.end();
}

template <typename T, typename U> bool contains(const std::unordered_set<T>& set, const U& value) {
    return set.find(value) != set.end();
}

template <typename T, typename U> bool containsByMember(const std::vector<T>& vector, U T::*member, const U& value) {
    return indexOfByMember(vector, member, value) != vector.size();
}

template <typename T, typename F> bool containsIf(const std::vector<T>& vector, const F& condition) {
    return std::find_if(vector.begin(), vector.end(), condition) != vector.end();
}

template <typename T, typename F> void eraseIf(std::vector<T>& vector, const F& condition) {
    vector.erase(std::remove_if(vector.begin(), vector.end(), condition), vector.end());
}

template <typename T, typename F> uint64_t countIf(const std::vector<T>& vector, const F& condition) {
    return static_cast<uint64_t>(std::count_if(vector.begin(), vector.end(), condition));
}

template <typename T> const T& getElementByIndex(const std::set<T>& set, uint64_t index) {
    assert(index < set.size());
    return *std::next(set.begin(), static_cast<int>(index));
}

template <typename T, typename F> void sort(std::vector<T>& vector, const F& comparison) {
    std::sort(vector.begin(), vector.end(), comparison);
}

template <typename T> void append(std::vector<T>& vector, const std::vector<T>& append) {
    vector.insert(vector.end(), append.begin(), append.end());
}

template <typename T, typename F> constexpr auto hasMemberImpl(F&& f) -> decltype(f(std::declval<T>()), true) {
    return true;
}

template <typename> constexpr bool hasMemberImpl(...) {
    return false;
}

#define HAS_MEMBER(T, EXPR) CppUtil::hasMemberImpl<T>([](auto&& obj) -> decltype(obj.EXPR) {})

} // namespace cesium::omniverse::CppUtil
