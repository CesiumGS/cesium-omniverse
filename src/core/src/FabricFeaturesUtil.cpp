#include "cesium/omniverse/FabricFeaturesUtil.h"

#include "cesium/omniverse/FabricFeaturesInfo.h"

namespace cesium::omniverse::FabricFeaturesUtil {

FabricFeatureIdType getFeatureIdType(const FabricFeatureId& featureId) {
    if (std::holds_alternative<std::monostate>(featureId.featureIdStorage)) {
        return FabricFeatureIdType::INDEX;
    } else if (std::holds_alternative<uint64_t>(featureId.featureIdStorage)) {
        return FabricFeatureIdType::ATTRIBUTE;
    } else if (std::holds_alternative<FabricTextureInfo>(featureId.featureIdStorage)) {
        return FabricFeatureIdType::TEXTURE;
    }

    return FabricFeatureIdType::INDEX;
}

std::vector<FabricFeatureIdType> getFeatureIdTypes(const FabricFeaturesInfo& featuresInfo) {
    const auto& featureIds = featuresInfo.featureIds;

    std::vector<FabricFeatureIdType> featureIdTypes;
    featureIdTypes.reserve(featureIds.size());

    for (const auto& featureId : featureIds) {
        featureIdTypes.push_back(getFeatureIdType(featureId));
    }

    return featureIdTypes;
}

std::vector<uint64_t> getSetIndexMapping(const FabricFeaturesInfo& featuresInfo, FabricFeatureIdType type) {
    const auto& featureIds = featuresInfo.featureIds;

    std::vector<uint64_t> setIndexMapping;
    setIndexMapping.reserve(featureIds.size());

    for (uint64_t i = 0; i < featureIds.size(); ++i) {
        if (getFeatureIdType(featureIds[i]) == type) {
            setIndexMapping.push_back(i);
        }
    }

    return setIndexMapping;
}

bool hasFeatureIdType(const FabricFeaturesInfo& featuresInfo, FabricFeatureIdType type) {
    const auto& featureIds = featuresInfo.featureIds;

    for (const auto& featureId : featureIds) {
        if (getFeatureIdType(featureId) == type) {
            return true;
        }
    }

    return false;
}

} // namespace cesium::omniverse::FabricFeaturesUtil
