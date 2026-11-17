#pragma once

#include <cstdint>
#include <vector>

namespace cesium::omniverse {
struct FeatureId;
struct FeaturesInfo;
enum class FeatureIdType;
} // namespace cesium::omniverse

namespace cesium::omniverse::FeaturesUtil {

FeatureIdType getFeatureIdType(const FeatureId& featureId);
std::vector<FeatureIdType> getFeatureIdTypes(const FeaturesInfo& featuresInfo);
std::vector<uint64_t> getSetIndexMapping(const FeaturesInfo& featuresInfo, FeatureIdType type);
bool hasFeatureIdType(const FeaturesInfo& featuresInfo, FeatureIdType type);

} // namespace cesium::omniverse::FeaturesUtil
