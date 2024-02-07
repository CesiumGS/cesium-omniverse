#pragma once

#include <cstdint>
#include <vector>

namespace cesium::omniverse {
struct FabricFeatureId;
struct FabricFeaturesInfo;
enum class FabricFeatureIdType;
} // namespace cesium::omniverse

namespace cesium::omniverse::FabricFeaturesUtil {

FabricFeatureIdType getFeatureIdType(const FabricFeatureId& featureId);
std::vector<FabricFeatureIdType> getFeatureIdTypes(const FabricFeaturesInfo& featuresInfo);
std::vector<uint64_t> getSetIndexMapping(const FabricFeaturesInfo& featuresInfo, FabricFeatureIdType type);
bool hasFeatureIdType(const FabricFeaturesInfo& featuresInfo, FabricFeatureIdType type);

} // namespace cesium::omniverse::FabricFeaturesUtil
