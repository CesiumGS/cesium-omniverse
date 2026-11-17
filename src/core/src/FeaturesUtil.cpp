#include "cesium/omniverse/FeaturesUtil.h"

#include "cesium/omniverse/FeaturesInfo.h"

namespace cesium::omniverse::FeaturesUtil {

FeatureIdType getFeatureIdType(const FeatureId& featureId) {
    if (std::holds_alternative<std::monostate>(featureId.featureIdStorage)) {
        return FeatureIdType::INDEX;
    } else if (std::holds_alternative<uint64_t>(featureId.featureIdStorage)) {
        return FeatureIdType::ATTRIBUTE;
    } else if (std::holds_alternative<TextureInfo>(featureId.featureIdStorage)) {
        return FeatureIdType::TEXTURE;
    }

    return FeatureIdType::INDEX;
}

std::vector<FeatureIdType> getFeatureIdTypes(const FeaturesInfo& featuresInfo) {
    const auto& featureIds = featuresInfo.featureIds;

    std::vector<FeatureIdType> featureIdTypes;
    featureIdTypes.reserve(featureIds.size());

    for (const auto& featureId : featureIds) {
        featureIdTypes.push_back(getFeatureIdType(featureId));
    }

    return featureIdTypes;
}

std::vector<uint64_t> getSetIndexMapping(const FeaturesInfo& featuresInfo, FeatureIdType type) {
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

bool hasFeatureIdType(const FeaturesInfo& featuresInfo, FeatureIdType type) {
    const auto& featureIds = featuresInfo.featureIds;

    for (const auto& featureId : featureIds) {
        if (getFeatureIdType(featureId) == type) {
            return true;
        }
    }

    return false;
}

} // namespace cesium::omniverse::FeaturesUtil
