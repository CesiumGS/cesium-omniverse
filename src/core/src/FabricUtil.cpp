#include "cesium/omniverse/FabricUtil.h"

#include "cesium/omniverse/DataType.h"
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
    omni::fabric::TokenC token;

  public:
    friend std::ostream& operator<<(std::ostream& os, const TokenWrapper& tokenWrapper);
};

std::ostream& operator<<(std::ostream& os, const TokenWrapper& tokenWrapper) {
    os << omni::fabric::Token(tokenWrapper.token).getString();
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
    omni::fabric::AssetPath asset;

  public:
    friend std::ostream& operator<<(std::ostream& os, const AssetWrapper& assetWrapper);
};

std::ostream& operator<<(std::ostream& os, const AssetWrapper& assetWrapper) {
    if (assetWrapper.asset.assetPath.IsEmpty()) {
        os << NO_DATA_STRING;
        return os;
    }

    os << "Asset Path: " << assetWrapper.asset.assetPath.GetText()
       << ", Resolved Path: " << assetWrapper.asset.resolvedPath.GetText();
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
std::string printAttributeValue(
    const omni::fabric::Path& primPath,
    const omni::fabric::Token& attributeName,
    const omni::fabric::AttributeRole& role) {

    using ElementType = std::array<BaseType, ComponentCount>;

    auto stageReaderWriter = UsdUtil::getFabricStage();

    if constexpr (IsArray) {
        const auto values = stageReaderWriter.getArrayAttributeRd<ElementType>(primPath, attributeName);
        const auto elementCount = values.size();

        if (elementCount == 0) {
            return NO_DATA_STRING;
        }

        const auto valuesPtr = values.front().data();

        if (role == omni::fabric::AttributeRole::eText) {
            return std::string(reinterpret_cast<const char*>(valuesPtr), elementCount);
        }

        return printAttributeValue<BaseType>(valuesPtr, elementCount, ComponentCount, true);
    } else {
        const auto value = stageReaderWriter.getAttributeRd<ElementType>(primPath, attributeName);

        if (value == nullptr) {
            return NO_DATA_STRING;
        }

        return printAttributeValue<BaseType>(value->data(), 1, ComponentCount, false);
    }
}

std::string printConnection(const omni::fabric::Path& primPath, const omni::fabric::Token& attributeName) {
    auto stageReaderWriter = UsdUtil::getFabricStage();
    const auto connection = stageReaderWriter.getConnection(primPath, attributeName);
    if (connection == nullptr) {
        return NO_DATA_STRING;
    }

    const auto path = omni::fabric::Path(connection->path).getText();
    const auto attrName = omni::fabric::Token(connection->attrName).getText();

    return fmt::format("Path: {}, Attribute Name: {}", path, attrName);
}

std::string printAttributeValue(const omni::fabric::Path& primPath, const omni::fabric::AttrNameAndType& attribute) {
    auto stageReaderWriter = UsdUtil::getFabricStage();

    const auto attributeType = attribute.type;
    const auto baseType = attributeType.baseType;
    const auto componentCount = attributeType.componentCount;
    const auto name = attribute.name;
    const auto arrayDepth = attributeType.arrayDepth;
    const auto role = attributeType.role;

    // This switch statement should cover most of the attribute types we expect to see on the stage.
    // This includes the USD types in SdfValueTypeNames and Fabric types like assets and tokens.
    // We can add more as needed.
    if (arrayDepth == 0) {
        switch (baseType) {
            case omni::fabric::BaseDataType::eAsset: {
                switch (componentCount) {
                    case 1: {
                        return printAttributeValue<false, AssetWrapper, 1>(primPath, name, role);
                    }
                    default: {
                        break;
                    }
                }
                break;
            }
            case omni::fabric::BaseDataType::eConnection: {
                switch (componentCount) {
                    case 1: {
                        return printConnection(primPath, name);
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
                        return printAttributeValue<false, TokenWrapper, 1>(primPath, name, role);
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
                        return printAttributeValue<false, BoolWrapper, 1>(primPath, name, role);
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
                        return printAttributeValue<false, uint8_t, 1>(primPath, name, role);
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
                        return printAttributeValue<false, int32_t, 1>(primPath, name, role);
                    }
                    case 2: {
                        return printAttributeValue<false, int32_t, 2>(primPath, name, role);
                    }
                    case 3: {
                        return printAttributeValue<false, int32_t, 3>(primPath, name, role);
                    }
                    case 4: {
                        return printAttributeValue<false, int32_t, 4>(primPath, name, role);
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
                        return printAttributeValue<false, uint32_t, 1>(primPath, name, role);
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
                        return printAttributeValue<false, int64_t, 1>(primPath, name, role);
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
                        return printAttributeValue<false, uint64_t, 1>(primPath, name, role);
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
                        return printAttributeValue<false, float, 1>(primPath, name, role);
                    }
                    case 2: {
                        return printAttributeValue<false, float, 2>(primPath, name, role);
                    }
                    case 3: {
                        return printAttributeValue<false, float, 3>(primPath, name, role);
                    }
                    case 4: {
                        return printAttributeValue<false, float, 4>(primPath, name, role);
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
                        return printAttributeValue<false, double, 1>(primPath, name, role);
                    }
                    case 2: {
                        return printAttributeValue<false, double, 2>(primPath, name, role);
                    }
                    case 3: {
                        return printAttributeValue<false, double, 3>(primPath, name, role);
                    }
                    case 4: {
                        return printAttributeValue<false, double, 4>(primPath, name, role);
                    }
                    case 6: {
                        return printAttributeValue<false, double, 6>(primPath, name, role);
                    }
                    case 9: {
                        return printAttributeValue<false, double, 9>(primPath, name, role);
                    }
                    case 16: {
                        return printAttributeValue<false, double, 16>(primPath, name, role);
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
                        return printAttributeValue<true, uint64_t, 1>(primPath, name, role);
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
                        return printAttributeValue<true, AssetWrapper, 1>(primPath, name, role);
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
                        return printAttributeValue<true, TokenWrapper, 1>(primPath, name, role);
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
                        return printAttributeValue<true, BoolWrapper, 1>(primPath, name, role);
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
                        return printAttributeValue<true, uint8_t, 1>(primPath, name, role);
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
                        return printAttributeValue<true, int32_t, 1>(primPath, name, role);
                    }
                    case 2: {
                        return printAttributeValue<true, int32_t, 2>(primPath, name, role);
                    }
                    case 3: {
                        return printAttributeValue<true, int32_t, 3>(primPath, name, role);
                    }
                    case 4: {
                        return printAttributeValue<true, int32_t, 4>(primPath, name, role);
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
                        return printAttributeValue<true, uint32_t, 1>(primPath, name, role);
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
                        return printAttributeValue<true, int64_t, 1>(primPath, name, role);
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
                        return printAttributeValue<true, uint64_t, 1>(primPath, name, role);
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
                        return printAttributeValue<true, float, 1>(primPath, name, role);
                    }
                    case 2: {
                        return printAttributeValue<true, float, 2>(primPath, name, role);
                    }
                    case 3: {
                        return printAttributeValue<true, float, 3>(primPath, name, role);
                    }
                    case 4: {
                        return printAttributeValue<true, float, 4>(primPath, name, role);
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
                        return printAttributeValue<true, double, 1>(primPath, name, role);
                    }
                    case 2: {
                        return printAttributeValue<true, double, 2>(primPath, name, role);
                    }
                    case 3: {
                        return printAttributeValue<true, double, 3>(primPath, name, role);
                    }
                    case 4: {
                        return printAttributeValue<true, double, 4>(primPath, name, role);
                    }
                    case 6: {
                        return printAttributeValue<true, double, 6>(primPath, name, role);
                    }
                    case 9: {
                        return printAttributeValue<true, double, 9>(primPath, name, role);
                    }
                    case 16: {
                        return printAttributeValue<true, double, 16>(primPath, name, role);
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

    auto stageReaderWriter = UsdUtil::getFabricStage();

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

    auto srw = UsdUtil::getFabricStage();

    const auto geometryBuckets = srw.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {omni::fabric::AttrNameAndType(FabricTypes::Mesh, FabricTokens::Mesh)});

    const auto materialBuckets = srw.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {omni::fabric::AttrNameAndType(FabricTypes::Material, FabricTokens::Material)});

    for (size_t bucketId = 0; bucketId < geometryBuckets.bucketCount(); bucketId++) {
        auto paths = srw.getPathArray(geometryBuckets, bucketId);

        // clang-format off
        auto worldVisibilityFabric = srw.getAttributeArrayRd<bool>(geometryBuckets, bucketId, FabricTokens::_worldVisibility);
        auto faceVertexCountsFabric = srw.getArrayAttributeArrayRd<int>(geometryBuckets, bucketId, FabricTokens::faceVertexCounts);
        auto tilesetIdFabric = srw.getAttributeArrayRd<int64_t>(geometryBuckets, bucketId, FabricTokens::_cesium_tilesetId);
        // clang-format on

        statistics.geometriesCapacity += paths.size();

        for (size_t i = 0; i < paths.size(); i++) {
            if (tilesetIdFabric[i] == NO_TILESET_ID) {
                continue;
            }

            statistics.geometriesLoaded++;

            const auto triangleCount = faceVertexCountsFabric[i].size();
            statistics.trianglesLoaded += triangleCount;

            if (worldVisibilityFabric[i]) {
                statistics.geometriesRendered++;
                statistics.trianglesRendered += triangleCount;
            }
        }
    }

    for (size_t bucketId = 0; bucketId < materialBuckets.bucketCount(); bucketId++) {
        auto paths = srw.getPathArray(materialBuckets, bucketId);

        // clang-format off
        auto tilesetIdFabric = srw.getAttributeArrayRd<int64_t>(materialBuckets, bucketId, FabricTokens::_cesium_tilesetId);
        // clang-format on

        statistics.materialsCapacity += paths.size();

        for (size_t i = 0; i < paths.size(); i++) {
            if (tilesetIdFabric[i] == NO_TILESET_ID) {
                continue;
            }

            statistics.materialsLoaded++;
        }
    }

    return statistics;
}

void destroyPrim(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStage();
    srw.destroyPrim(path);

    // Prims removed from Fabric need special handling for their removal to be reflected in the Hydra render index
    // This workaround may not be needed in future Kit versions, but is needed as of Kit 105.0
    const omni::fabric::Path changeTrackingPath("/TempChangeTracking");

    if (srw.getAttributeRd<omni::fabric::PathC>(changeTrackingPath, FabricTokens::_deletedPrims) == nullptr) {
        return;
    }

    const auto deletedPrimsSize = srw.getArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims);
    srw.setArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims, deletedPrimsSize + 1);
    auto deletedPrimsFabric =
        srw.getArrayAttributeWr<omni::fabric::PathC>(changeTrackingPath, FabricTokens::_deletedPrims);

    deletedPrimsFabric[deletedPrimsSize] = path;
}

void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToUsdTransform) {
    auto srw = UsdUtil::getFabricStage();

    const auto buckets = srw.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId)},
        {omni::fabric::AttrNameAndType(
            FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform)});

    for (size_t bucketId = 0; bucketId < buckets.bucketCount(); bucketId++) {
        // clang-format off
        auto tilesetIdFabric = srw.getAttributeArrayRd<int64_t>(buckets, bucketId, FabricTokens::_cesium_tilesetId);
        auto localToEcefTransformFabric = srw.getAttributeArrayRd<pxr::GfMatrix4d>(buckets, bucketId, FabricTokens::_cesium_localToEcefTransform);
        auto extentFabric = srw.getAttributeArrayRd<pxr::GfRange3d>(buckets, bucketId, FabricTokens::extent);

        auto worldPositionFabric = srw.getAttributeArrayWr<pxr::GfVec3d>(buckets, bucketId, FabricTokens::_worldPosition);
        auto worldOrientationFabric = srw.getAttributeArrayWr<pxr::GfQuatf>(buckets, bucketId, FabricTokens::_worldOrientation);
        auto worldScaleFabric = srw.getAttributeArrayWr<pxr::GfVec3f>(buckets, bucketId, FabricTokens::_worldScale);
        auto worldExtentFabric = srw.getAttributeArrayWr<pxr::GfRange3d>(buckets, bucketId, FabricTokens::_worldExtent);
        // clang-format on

        for (size_t i = 0; i < tilesetIdFabric.size(); i++) {
            if (tilesetIdFabric[i] == tilesetId) {
                const auto localToEcefTransform = UsdUtil::usdToGlmMatrix(localToEcefTransformFabric[i]);
                const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
                const auto extent = extentFabric[i];
                const auto [worldPosition, worldOrientation, worldScale] =
                    UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);
                const auto worldExtent = UsdUtil::computeWorldExtent(extent, localToUsdTransform);

                worldPositionFabric[i] = worldPosition;
                worldOrientationFabric[i] = worldOrientation;
                worldScaleFabric[i] = worldScale;
                worldExtentFabric[i] = worldExtent;
            }
        }
    }
}

void setTilesetId(const omni::fabric::Path& path, int64_t tilesetId) {
    auto srw = UsdUtil::getFabricStage();

    auto tilesetIdFabric = srw.getAttributeWr<int64_t>(path, FabricTokens::_cesium_tilesetId);

    *tilesetIdFabric = tilesetId;
}

omni::fabric::Path toFabricPath(const pxr::SdfPath& path) {
    return {omni::fabric::asInt(path)};
}

omni::fabric::Token toFabricToken(const pxr::TfToken& token) {
    return {omni::fabric::asInt(token)};
}

omni::fabric::Path joinPaths(const omni::fabric::Path& absolutePath, const omni::fabric::Token& relativePath) {
    return {fmt::format("{}/{}", absolutePath.getText(), relativePath.getText()).c_str()};
}

omni::fabric::Path getCopiedShaderPath(const omni::fabric::Path& materialPath, const omni::fabric::Path& shaderPath) {
    // materialPath is the FabricMaterial path
    // shaderPath is the USD shader path
    return FabricUtil::joinPaths(materialPath, omni::fabric::Token(UsdUtil::getSafeName(shaderPath.getText()).c_str()));
}

namespace {

struct FabricConnection {
    omni::fabric::Connection* connection;
    omni::fabric::Token attributeName;
};

std::vector<FabricConnection> getConnections(const omni::fabric::Path& path) {
    std::vector<FabricConnection> connections;

    auto srw = UsdUtil::getFabricStage();
    const auto attributes = srw.getAttributeNamesAndTypes(path);
    const auto& names = attributes.first;
    const auto& types = attributes.second;

    for (size_t i = 0; i < names.size(); i++) {
        const auto& name = names[i];
        const auto& type = types[i];
        if (type.baseType == omni::fabric::BaseDataType::eConnection) {
            const auto connection = srw.getConnection(path, name);
            if (connection != nullptr) {
                connections.emplace_back(FabricConnection{connection, name});
            }
        }
    }

    return connections;
}

bool isOutput(const omni::fabric::Token& attributeName) {
    return attributeName == FabricTokens::outputs_out;
}

bool isConnection(const omni::fabric::Type& attributeType) {
    return attributeType.baseType == omni::fabric::BaseDataType::eConnection;
}

bool isEmptyToken(
    const omni::fabric::Path& path,
    const omni::fabric::Token& attributeName,
    const omni::fabric::Type& attributeType) {
    auto srw = UsdUtil::getFabricStage();
    if (attributeType.baseType == omni::fabric::BaseDataType::eToken) {
        const auto attributeValue = srw.getAttributeRd<omni::fabric::Token>(path, attributeName);
        if (attributeValue == nullptr || (*attributeValue).size() == 0) {
            return true;
        }
    }

    return false;
}

std::vector<omni::fabric::TokenC> getAttributesToCopy(const omni::fabric::Path& path) {
    std::vector<omni::fabric::TokenC> attributeNames;

    auto srw = UsdUtil::getFabricStage();

    const auto attributes = srw.getAttributeNamesAndTypes(path);
    const auto& names = attributes.first;
    const auto& types = attributes.second;

    for (size_t i = 0; i < names.size(); i++) {
        const auto& name = names[i];
        const auto& type = types[i];

        if (!isOutput(name) && !isConnection(type) && !isEmptyToken(path, name, type)) {
            attributeNames.emplace_back(omni::fabric::TokenC(name));
        }
    }

    return attributeNames;
}

struct FabricAttribute {
    omni::fabric::Token name;
    omni::fabric::Type type;
};

std::vector<FabricAttribute> getAttributesToCreate(const omni::fabric::Path& path) {
    std::vector<FabricAttribute> attributeNames;

    auto srw = UsdUtil::getFabricStage();

    const auto attributes = srw.getAttributeNamesAndTypes(path);
    const auto& names = attributes.first;
    const auto& types = attributes.second;

    for (size_t i = 0; i < names.size(); i++) {
        const auto& name = names[i];
        const auto& type = types[i];

        if (isOutput(name) || isEmptyToken(path, name, type)) {
            attributeNames.emplace_back(FabricAttribute{name, type});
        }
    }

    return attributeNames;
}

void getConnectedPrimsRecursive(const omni::fabric::Path& path, std::vector<omni::fabric::Path>& connectedPaths) {
    const auto connections = getConnections(path);
    for (const auto& connection : connections) {
        const auto it = std::find(connectedPaths.begin(), connectedPaths.end(), connection.connection->path);
        if (it == connectedPaths.end()) {
            connectedPaths.emplace_back(connection.connection->path);
            getConnectedPrimsRecursive(connection.connection->path, connectedPaths);
        }
    }
}

std::vector<omni::fabric::Path> getPrimsInMaterialNetwork(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStage();
    std::vector<omni::fabric::Path> paths;
    paths.push_back(path);
    getConnectedPrimsRecursive(path, paths);
    return paths;
}

} // namespace

std::vector<omni::fabric::Path>
copyMaterial(const omni::fabric::Path& srcMaterialPath, const omni::fabric::Path& dstMaterialPath) {
    auto srw = UsdUtil::getFabricStage();
    const auto isrw = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();

    const auto srcPaths = getPrimsInMaterialNetwork(srcMaterialPath);

    std::vector<omni::fabric::Path> dstPaths;
    dstPaths.reserve(srcPaths.size());

    for (const auto& srcPath : srcPaths) {
        auto dstPath = omni::fabric::Path();

        if (srcPath == srcMaterialPath) {
            dstPath = dstMaterialPath;
        } else {
            dstPath = FabricUtil::getCopiedShaderPath(dstMaterialPath, srcPath);
        }

        dstPaths.push_back(dstPath);

        srw.createPrim(dstPath);

        // This excludes connections, outputs, and empty tokens
        // The material network will be reconnected later once all the prims have been copied
        // The reason for excluding outputs and empty tokens is so that Omniverse doesn't print the warning
        //   [Warning] [omni.fabric.plugin] Warning: input has no valid data
        const auto attributesToCopy = getAttributesToCopy(srcPath);

        isrw->copySpecifiedAttributes(
            srw.getId(), srcPath, attributesToCopy.data(), dstPath, attributesToCopy.data(), attributesToCopy.size());

        // Add the outputs and empty tokens back. This doesn't print a warning.
        const auto attributesToCreate = getAttributesToCreate(srcPath);
        for (const auto& attribute : attributesToCreate) {
            srw.createAttribute(dstPath, attribute.name, attribute.type);
        }
    }

    // Reconnect the prims
    for (size_t i = 0; i < srcPaths.size(); i++) {
        const auto& srcPath = srcPaths[i];
        const auto& dstPath = dstPaths[i];
        const auto connections = getConnections(srcPath);
        for (const auto& connection : connections) {
            const auto it = std::find(srcPaths.begin(), srcPaths.end(), connection.connection->path);
            assert(it != srcPaths.end()); // Ensure that all connections are part of the material network
            const auto index = it - srcPaths.begin();
            const auto dstConnection =
                omni::fabric::Connection{omni::fabric::PathC(dstPaths[index]), connection.connection->attrName};
            srw.createConnection(dstPath, connection.attributeName, dstConnection);
        }
    }

    return dstPaths;
}

bool materialHasCesiumNodes(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStage();
    const auto paths = getPrimsInMaterialNetwork(path);

    for (const auto& p : paths) {
        const auto mdlIdentifier = getMdlIdentifier(p);
        if (isCesiumNode(mdlIdentifier)) {
            return true;
        }
    }

    return false;
}

bool isCesiumNode(const omni::fabric::Token& mdlIdentifier) {
    return mdlIdentifier == FabricTokens::cesium_base_color_texture_float4 ||
           mdlIdentifier == FabricTokens::cesium_imagery_layer_float4 ||
           mdlIdentifier == FabricTokens::cesium_feature_id_int || isCesiumPropertyNode(mdlIdentifier);
}

bool isCesiumPropertyNode(const omni::fabric::Token& mdlIdentifier) {
    return mdlIdentifier == FabricTokens::cesium_property_int || mdlIdentifier == FabricTokens::cesium_property_int2 ||
           mdlIdentifier == FabricTokens::cesium_property_int3 || mdlIdentifier == FabricTokens::cesium_property_int4 ||
           mdlIdentifier == FabricTokens::cesium_property_float ||
           mdlIdentifier == FabricTokens::cesium_property_float2 ||
           mdlIdentifier == FabricTokens::cesium_property_float3 ||
           mdlIdentifier == FabricTokens::cesium_property_float4;
}

bool isShaderConnectedToMaterial(const omni::fabric::Path& materialPath, const omni::fabric::Path& shaderPath) {
    auto srw = UsdUtil::getFabricStage();
    const auto paths = getPrimsInMaterialNetwork(materialPath);
    return std::find(paths.begin(), paths.end(), shaderPath) != paths.end();
}

omni::fabric::Token getMdlIdentifier(const omni::fabric::Path& path) {
    auto srw = UsdUtil::getFabricStage();
    if (srw.attributeExists(path, FabricTokens::info_mdl_sourceAsset_subIdentifier)) {
        const auto infoMdlSourceAssetSubIdentifierFabric =
            srw.getAttributeRd<omni::fabric::Token>(path, FabricTokens::info_mdl_sourceAsset_subIdentifier);
        if (infoMdlSourceAssetSubIdentifierFabric != nullptr) {
            return *infoMdlSourceAssetSubIdentifierFabric;
        }
    }
    return omni::fabric::Token{};
}

omni::fabric::Type getPrimvarType(DataType type) {
    const auto baseDataType = getPrimvarBaseDataType(type);
    const auto componentCount = getComponentCount(type);
    return {baseDataType, static_cast<uint8_t>(componentCount), 1, omni::fabric::AttributeRole::eNone};
}

MdlExternalPropertyType getMdlExternalPropertyType(const omni::fabric::Token& mdlIdentifier) {
    assert(isCesiumPropertyNode(mdlIdentifier));

    if (mdlIdentifier == FabricTokens::cesium_property_int) {
        return MdlExternalPropertyType::INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_int2) {
        return MdlExternalPropertyType::VEC2_INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_int3) {
        return MdlExternalPropertyType::VEC3_INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_int4) {
        return MdlExternalPropertyType::VEC4_INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float) {
        return MdlExternalPropertyType::FLOAT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float2) {
        return MdlExternalPropertyType::VEC2_FLOAT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float3) {
        return MdlExternalPropertyType::VEC3_FLOAT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float4) {
        return MdlExternalPropertyType::VEC4_FLOAT32;
    }

    // Should never reach here
    assert(false);
    return MdlExternalPropertyType::INT32;
}

bool typesCompatible(MdlExternalPropertyType externalType, MdlInternalPropertyType internalType) {
    switch (externalType) {
        case MdlExternalPropertyType::INT32:
            switch (internalType) {
                case MdlInternalPropertyType::INT32:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::FLOAT32:
                case MdlInternalPropertyType::INT32_NORM:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::VEC2_INT32:
            switch (internalType) {
                case MdlInternalPropertyType::VEC2_INT32:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::VEC2_FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::VEC2_FLOAT32:
                case MdlInternalPropertyType::VEC2_INT32_NORM:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::VEC3_INT32:
            switch (internalType) {
                case MdlInternalPropertyType::VEC3_INT32:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::VEC3_FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::VEC3_FLOAT32:
                case MdlInternalPropertyType::VEC3_INT32_NORM:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::VEC4_INT32:
            switch (internalType) {
                case MdlInternalPropertyType::VEC4_INT32:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::VEC4_FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::VEC4_FLOAT32:
                case MdlInternalPropertyType::VEC4_INT32_NORM:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::MAT2_FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::MAT2_INT32:
                case MdlInternalPropertyType::MAT2_FLOAT32:
                case MdlInternalPropertyType::MAT2_INT32_NORM:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::MAT3_FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::MAT3_INT32:
                case MdlInternalPropertyType::MAT3_FLOAT32:
                case MdlInternalPropertyType::MAT3_INT32_NORM:
                    return true;
                default:
                    return false;
            }
        case MdlExternalPropertyType::MAT4_FLOAT32:
            switch (internalType) {
                case MdlInternalPropertyType::MAT4_INT32:
                case MdlInternalPropertyType::MAT4_FLOAT32:
                case MdlInternalPropertyType::MAT4_INT32_NORM:
                    return true;
                default:
                    return false;
            }
    }

    // Shouldn't reach here. All cases handled above.
    assert(false);
    return false;
}

} // namespace cesium::omniverse::FabricUtil
