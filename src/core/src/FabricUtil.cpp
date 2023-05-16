#include "cesium/omniverse/FabricUtil.h"

#include "cesium/omniverse/FabricAsset.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <omni/fabric/FabricUSD.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <spdlog/fmt/fmt.h>

#include <sstream>

namespace cesium::omniverse::FabricUtil {

namespace {

const char* const NO_DATA_STRING = "[No Data]";
const char* const TYPE_NOT_SUPPORTED_STRING = "[Type Not Supported]";

// Wraps the token type so that we can define a custom stream insertion operator
class TokenWrapper {
  private:
    omni::fabric::Token token;

  public:
    friend std::ostream& operator<<(std::ostream& os, const TokenWrapper& tokenWrapper);
};

std::ostream& operator<<(std::ostream& os, const TokenWrapper& tokenWrapper) {
    os << tokenWrapper.token.getString();
    return os;
}

// Wraps a boolean so that we print "true" and "false" instead of 0 and 1
class BoolWrapper {
  private:
    bool value;

  public:
    friend std::ostream& operator<<(std::ostream& os, const BoolWrapper& boolWrapper);
};

std::ostream& operator<<(std::ostream& os, const BoolWrapper& boolWrapper) {
    os << (boolWrapper.value ? "true" : "false");
    return os;
}

class AssetWrapper {
  private:
    FabricAsset asset;

  public:
    friend std::ostream& operator<<(std::ostream& os, const AssetWrapper& assetWrapper);
};

std::ostream& operator<<(std::ostream& os, const AssetWrapper& assetWrapper) {
    if (assetWrapper.asset.isEmpty()) {
        os << NO_DATA_STRING;
        return os;
    }

    os << "Asset Path: " << assetWrapper.asset.getAssetPath()
       << ", Resolved Path: " << assetWrapper.asset.getResolvedPath();
    return os;
}

template <typename T>
std::string printAttributeValue(const T* values, uint64_t elementCount, uint64_t componentCount, bool isArray) {
    std::stringstream stream;

    if (isArray) {
        stream << "[";
    }

    for (uint64_t i = 0; i < elementCount; i++) {
        if (componentCount > 1) {
            stream << "[";
        }

        for (uint64_t j = 0; j < componentCount; j++) {
            stream << values[i * componentCount + j];
            if (j < componentCount - 1) {
                stream << ",";
            }
        }

        if (componentCount > 1) {
            stream << "]";
        }

        if (elementCount > 1 && i < elementCount - 1) {
            stream << ",";
        }
    }

    if (isArray) {
        stream << "]";
    }

    return stream.str();
}

template <bool IsArray, typename BaseType, uint64_t ComponentCount>
std::string printAttributeValue(const omni::fabric::Path& primPath, const omni::fabric::Token& attributeName) {
    using ElementType = std::array<BaseType, ComponentCount>;

    auto stageReaderWriter = UsdUtil::getFabricStageReaderWriter();

    if constexpr (IsArray) {
        const auto values = stageReaderWriter.getArrayAttributeRd<ElementType>(primPath, attributeName);
        const auto elementCount = values.size();

        if (elementCount == 0) {
            return NO_DATA_STRING;
        }

        return printAttributeValue<BaseType>(values.front().data(), elementCount, ComponentCount, true);
    } else {
        const auto value = stageReaderWriter.getAttributeRd<ElementType>(primPath, attributeName);

        if (value == nullptr) {
            return NO_DATA_STRING;
        }

        return printAttributeValue<BaseType>(value->data(), 1, ComponentCount, false);
    }
}

std::string printAttributeValue(const omni::fabric::Path& primPath, const omni::fabric::AttrNameAndType& attribute) {
    auto stageReaderWriter = UsdUtil::getFabricStageReaderWriter();

    const auto attributeType = attribute.type;
    const auto baseType = attributeType.baseType;
    const auto componentCount = attributeType.componentCount;
    const auto name = attribute.name;
    const auto arrayDepth = attributeType.arrayDepth;

    // This switch statement should cover most of the attribute types we expect to see on the stage.
    // This includes the USD types in SdfValueTypeNames and Fabric types like assets and tokens.
    // We can add more as needed.
    if (arrayDepth == 0) {
        switch (baseType) {
            case omni::fabric::BaseDataType::eAsset: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, AssetWrapper, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eToken: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, TokenWrapper, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eBool: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, BoolWrapper, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eUChar: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, uint8_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eInt: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, int32_t, 1>(primPath, name);
                    }
                    case 2: {
                        return printAttributeValue<false, int32_t, 2>(primPath, name);
                    }
                    case 3: {
                        return printAttributeValue<false, int32_t, 3>(primPath, name);
                    }
                    case 4: {
                        return printAttributeValue<false, int32_t, 4>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eUInt: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, uint32_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eInt64: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, int64_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eUInt64: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, uint64_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eFloat: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, float, 1>(primPath, name);
                    }
                    case 2: {
                        return printAttributeValue<false, float, 2>(primPath, name);
                    }
                    case 3: {
                        return printAttributeValue<false, float, 3>(primPath, name);
                    }
                    case 4: {
                        return printAttributeValue<false, float, 4>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eDouble: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, double, 1>(primPath, name);
                    }
                    case 2: {
                        return printAttributeValue<false, double, 2>(primPath, name);
                    }
                    case 3: {
                        return printAttributeValue<false, double, 3>(primPath, name);
                    }
                    case 4: {
                        return printAttributeValue<false, double, 4>(primPath, name);
                    }
                    case 6: {
                        return printAttributeValue<false, double, 6>(primPath, name);
                    }
                    case 9: {
                        return printAttributeValue<false, double, 9>(primPath, name);
                    }
                    case 16: {
                        return printAttributeValue<false, double, 16>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            // Due to legacy support the eRelationship type is defined as a scalar value but is secretly an array
            case omni::fabric::BaseDataType::eRelationship: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, int64_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            default: {
                break;
            }
        }
    } else if (arrayDepth == 1) {
        switch (baseType) {
            case omni::fabric::BaseDataType::eAsset: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, AssetWrapper, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eToken: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, TokenWrapper, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eBool: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, BoolWrapper, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eUChar: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, uint8_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eInt: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, int32_t, 1>(primPath, name);
                    }
                    case 2: {
                        return printAttributeValue<true, int32_t, 2>(primPath, name);
                    }
                    case 3: {
                        return printAttributeValue<true, int32_t, 3>(primPath, name);
                    }
                    case 4: {
                        return printAttributeValue<true, int32_t, 4>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eUInt: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, uint32_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eInt64: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, int64_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eUInt64: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, uint64_t, 1>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eFloat: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, float, 1>(primPath, name);
                    }
                    case 2: {
                        return printAttributeValue<true, float, 2>(primPath, name);
                    }
                    case 3: {
                        return printAttributeValue<true, float, 3>(primPath, name);
                    }
                    case 4: {
                        return printAttributeValue<true, float, 4>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eDouble: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<true, double, 1>(primPath, name);
                    }
                    case 2: {
                        return printAttributeValue<true, double, 2>(primPath, name);
                    }
                    case 3: {
                        return printAttributeValue<true, double, 3>(primPath, name);
                    }
                    case 4: {
                        return printAttributeValue<true, double, 4>(primPath, name);
                    }
                    case 6: {
                        return printAttributeValue<true, double, 6>(primPath, name);
                    }
                    case 9: {
                        return printAttributeValue<true, double, 9>(primPath, name);
                    }
                    case 16: {
                        return printAttributeValue<true, double, 16>(primPath, name);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            default: {
                break;
            }
        }
    }

    return TYPE_NOT_SUPPORTED_STRING;
}

} // namespace

std::string printFabricStage() {
    std::stringstream stream;

    auto stageReaderWriter = UsdUtil::getFabricStageReaderWriter();

    // For extra debugging. This gets printed to the console.
    stageReaderWriter.printBucketNames();

    // This returns ALL the buckets
    const auto& buckets = stageReaderWriter.findPrims({});

    for (size_t bucketId = 0; bucketId < buckets.bucketCount(); bucketId++) {
        const auto& attributes = stageReaderWriter.getAttributeNamesAndTypes(buckets, bucketId);
        const auto& primPaths = stageReaderWriter.getPathArray(buckets, bucketId);

        for (const auto& primPath : primPaths) {
            const auto primPathString = primPath.getText();
            const auto primPathUint64 = omni::fabric::PathC(primPath).path;

            stream << fmt::format("Prim: {} ({})\n", primPathString, primPathUint64);
            stream << fmt::format("  Attributes:\n");

            for (const auto& attribute : attributes) {
                const auto attributeName = attribute.name.getText();
                const auto attributeType = attribute.type.getTypeName();
                const auto attributeBaseType = attribute.type.baseType;
                const auto attributeValue = printAttributeValue(primPath, attribute);

                stream << fmt::format("    Attribute: {}\n", attributeName);
                stream << fmt::format("      Type: {}\n", attributeType);

                if (attributeBaseType != omni::fabric::BaseDataType::eTag) {
                    stream << fmt::format("      Value: {}\n", attributeValue);
                }
            }
        }
    }

    return stream.str();
}

FabricStatistics getStatistics() {
    FabricStatistics statistics;

    if (!UsdUtil::hasStage()) {
        return statistics;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto geometryBuckets = srw.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {omni::fabric::AttrNameAndType(FabricTypes::Mesh, FabricTokens::Mesh)});

    const auto materialBuckets = srw.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {omni::fabric::AttrNameAndType(FabricTypes::Material, FabricTokens::Material)});

    for (size_t bucketId = 0; bucketId < geometryBuckets.bucketCount(); bucketId++) {
        auto paths = srw.getPathArray(geometryBuckets, bucketId);
        statistics.numberOfGeometriesLoaded += paths.size();

        auto worldVisibilityFabric =
            srw.getAttributeArrayRd<bool>(geometryBuckets, bucketId, FabricTokens::_worldVisibility);
        statistics.numberOfGeometriesVisible +=
            std::count(worldVisibilityFabric.begin(), worldVisibilityFabric.end(), true);
    }

    for (size_t bucketId = 0; bucketId < materialBuckets.bucketCount(); bucketId++) {
        auto paths = srw.getPathArray(materialBuckets, bucketId);
        statistics.numberOfMaterialsLoaded += paths.size();
    }

    return statistics;
}

namespace {
void destroyPrimsSpan(gsl::span<const pxr::SdfPath> paths) {
    // Only delete prims if there's still a stage to delete them from
    if (!UsdUtil::hasStage()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();

    for (const auto& path : paths) {
        srw.destroyPrim(omni::fabric::asInt(path));
    }

    // Prims removed from Fabric need special handling for their removal to be reflected in the Hydra render index
    // This workaround may not be needed in future Kit versions, but is needed as of Kit 104.2
    const omni::fabric::Path changeTrackingPath("/TempChangeTracking");

    if (srw.getAttribute<uint64_t>(changeTrackingPath, FabricTokens::_deletedPrims) == nullptr) {
        return;
    }

    const auto deletedPrimsSize = srw.getArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims);
    srw.setArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims, deletedPrimsSize + paths.size());
    auto deletedPrimsFabric = srw.getArrayAttributeWr<uint64_t>(changeTrackingPath, FabricTokens::_deletedPrims);

    for (size_t i = 0; i < paths.size(); i++) {
        deletedPrimsFabric[deletedPrimsSize + i] = omni::fabric::asInt(paths[i]).path;
    }
}
} // namespace

void destroyPrim(const pxr::SdfPath& path) {
    destroyPrimsSpan(gsl::span(&path, 1));
}

void destroyPrims(const std::vector<pxr::SdfPath>& paths) {
    destroyPrimsSpan(gsl::span(paths));
}

void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto buckets = srw.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {omni::fabric::AttrNameAndType(
            FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform)});

    for (size_t bucketId = 0; bucketId < buckets.bucketCount(); bucketId++) {
        // clang-format off
        auto tilesetIdFabric = srw.getAttributeArrayRd<int64_t>(buckets, bucketId, FabricTokens::_cesium_tilesetId);
        auto localToEcefTransformFabric = srw.getAttributeArrayRd<pxr::GfMatrix4d>(buckets, bucketId, FabricTokens::_cesium_localToEcefTransform);
        auto localExtentFabric = srw.getAttributeArrayRd<pxr::GfRange3d>(buckets, bucketId, FabricTokens::_localExtent);

        auto worldPositionFabric = srw.getAttributeArrayWr<pxr::GfVec3d>(buckets, bucketId, FabricTokens::_worldPosition);
        auto worldOrientationFabric = srw.getAttributeArrayWr<pxr::GfQuatf>(buckets, bucketId, FabricTokens::_worldOrientation);
        auto worldScaleFabric = srw.getAttributeArrayWr<pxr::GfVec3f>(buckets, bucketId, FabricTokens::_worldScale);
        auto worldExtentFabric = srw.getAttributeArrayWr<pxr::GfRange3d>(buckets, bucketId, FabricTokens::_worldExtent);
        // clang-format on

        for (size_t i = 0; i < tilesetIdFabric.size(); i++) {
            if (tilesetIdFabric[i] == tilesetId) {
                const auto localToEcefTransform = UsdUtil::usdToGlmMatrix(localToEcefTransformFabric[i]);
                const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
                const auto localExtent = localExtentFabric[i];
                const auto [worldPosition, worldOrientation, worldScale] =
                    UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);
                const auto worldExtent = UsdUtil::computeWorldExtent(localExtent, localToUsdTransform);

                worldPositionFabric[i] = worldPosition;
                worldOrientationFabric[i] = worldOrientation;
                worldScaleFabric[i] = worldScale;
                worldExtentFabric[i] = worldExtent;
            }
        }
    }
}

void setTilesetIdAndTileId(const pxr::SdfPath& path, int64_t tilesetId, int64_t tileId) {
    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto pathFabric = omni::fabric::Path(omni::fabric::asInt(path));
    auto tilesetIdFabric = srw.getAttributeWr<int64_t>(pathFabric, FabricTokens::_cesium_tilesetId);
    auto tileIdFabric = srw.getAttributeWr<int64_t>(pathFabric, FabricTokens::_cesium_tileId);

    *tilesetIdFabric = tilesetId;
    *tileIdFabric = tileId;
}

} // namespace cesium::omniverse::FabricUtil
