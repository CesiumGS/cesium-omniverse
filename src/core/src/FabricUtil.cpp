#include "cesium/omniverse/FabricUtil.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/DataTypeUtil.h"
#include "cesium/omniverse/FabricStatistics.h"
#include "cesium/omniverse/MathUtil.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <fmt/format.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/SimStageWithHistory.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>

#include <sstream>

namespace cesium::omniverse::FabricUtil {

namespace {

const std::string_view NO_DATA_STRING = "[No Data]";
const std::string_view TYPE_NOT_SUPPORTED_STRING = "[Type Not Supported]";

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
    omni::fabric::AssetPath asset;

  public:
    friend std::ostream& operator<<(std::ostream& os, const AssetWrapper& assetWrapper);
};

std::ostream& operator<<(std::ostream& os, const AssetWrapper& assetWrapper) {
    if (assetWrapper.asset.assetPath.isNull()) {
        os << NO_DATA_STRING;
        return os;
    }

    os << "Asset Path: " << assetWrapper.asset.assetPath.getText()
       << ", Resolved Path: " << assetWrapper.asset.resolvedPath.getText();
    return os;
}

template <typename T>
std::string printAttributeValue(const T* values, uint64_t elementCount, uint64_t componentCount, bool isArray) {
    std::stringstream stream;

    if (isArray) {
        stream << "[";
    }

    for (uint64_t i = 0; i < elementCount; ++i) {
        if (componentCount > 1) {
            stream << "[";
        }

        for (uint64_t j = 0; j < componentCount; ++j) {
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
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& primPath,
    const omni::fabric::Token& attributeName,
    const omni::fabric::AttributeRole& role) {

    using ElementType = std::array<BaseType, ComponentCount>;

    if constexpr (IsArray) {
        const auto values = fabricStage.getArrayAttributeRd<ElementType>(primPath, attributeName);
        const auto elementCount = values.size();

        if (elementCount == 0) {
            return std::string(NO_DATA_STRING);
        }

        const auto valuesPtr = values.front().data();

        if (role == omni::fabric::AttributeRole::eText) {
            return std::string(reinterpret_cast<const char*>(valuesPtr), elementCount);
        }

        return printAttributeValue<BaseType>(valuesPtr, elementCount, ComponentCount, true);
    } else {
        const auto pValue = fabricStage.getAttributeRd<ElementType>(primPath, attributeName);

        if (!pValue) {
            return std::string(NO_DATA_STRING);
        }

        return printAttributeValue<BaseType>(pValue->data(), 1, ComponentCount, false);
    }
}

std::string printConnection(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& primPath,
    const omni::fabric::Token& attributeName) {
    const auto pConnection = fabricStage.getConnection(primPath, attributeName);
    if (!pConnection) {
        return std::string(NO_DATA_STRING);
    }

    const auto path = pConnection->path.getString();
    const auto attrName = pConnection->attrName.getText();

    return fmt::format("Path: {}, Attribute Name: {}", path, attrName);
}

std::string printAttributeValue(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& primPath,
    const omni::fabric::AttrNameAndType& attribute) {
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
            case omni::fabric::BaseDataType::eAsset:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, AssetWrapper, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eConnection:
                switch (componentCount) {
                    case 1:
                        return printConnection(fabricStage, primPath, name);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eToken:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, TokenWrapper, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eBool:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, BoolWrapper, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eUChar:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, uint8_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eInt:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, int32_t, 1>(fabricStage, primPath, name, role);
                    case 2:
                        return printAttributeValue<false, int32_t, 2>(fabricStage, primPath, name, role);
                    case 3:
                        return printAttributeValue<false, int32_t, 3>(fabricStage, primPath, name, role);
                    case 4:
                        return printAttributeValue<false, int32_t, 4>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eUInt:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, uint32_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eInt64:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, int64_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eUInt64:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, uint64_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eFloat:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, float, 1>(fabricStage, primPath, name, role);
                    case 2:
                        return printAttributeValue<false, float, 2>(fabricStage, primPath, name, role);
                    case 3:
                        return printAttributeValue<false, float, 3>(fabricStage, primPath, name, role);
                    case 4:
                        return printAttributeValue<false, float, 4>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eDouble:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<false, double, 1>(fabricStage, primPath, name, role);
                    case 2:
                        return printAttributeValue<false, double, 2>(fabricStage, primPath, name, role);
                    case 3:
                        return printAttributeValue<false, double, 3>(fabricStage, primPath, name, role);
                    case 4:
                        return printAttributeValue<false, double, 4>(fabricStage, primPath, name, role);
                    case 6:
                        return printAttributeValue<false, double, 6>(fabricStage, primPath, name, role);
                    case 9:
                        return printAttributeValue<false, double, 9>(fabricStage, primPath, name, role);
                    case 16:
                        return printAttributeValue<false, double, 16>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            // Due to legacy support the eRelationship type is defined as a scalar value but is secretly an array
            case omni::fabric::BaseDataType::eRelationship:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, uint64_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            default:
                break;
        }
    } else if (arrayDepth == 1) {
        switch (baseType) {
            case omni::fabric::BaseDataType::eAsset:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, AssetWrapper, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eToken:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, TokenWrapper, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eBool:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, BoolWrapper, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eUChar:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, uint8_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eInt:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, int32_t, 1>(fabricStage, primPath, name, role);
                    case 2:
                        return printAttributeValue<true, int32_t, 2>(fabricStage, primPath, name, role);
                    case 3:
                        return printAttributeValue<true, int32_t, 3>(fabricStage, primPath, name, role);
                    case 4:
                        return printAttributeValue<true, int32_t, 4>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eUInt:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, uint32_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eInt64:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, int64_t, 1>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eUInt64:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, uint64_t, 1>(fabricStage, primPath, name, role);

                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eFloat:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, float, 1>(fabricStage, primPath, name, role);
                    case 2:
                        return printAttributeValue<true, float, 2>(fabricStage, primPath, name, role);
                    case 3:
                        return printAttributeValue<true, float, 3>(fabricStage, primPath, name, role);
                    case 4:
                        return printAttributeValue<true, float, 4>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            case omni::fabric::BaseDataType::eDouble:
                switch (componentCount) {
                    case 1:
                        return printAttributeValue<true, double, 1>(fabricStage, primPath, name, role);
                    case 2:
                        return printAttributeValue<true, double, 2>(fabricStage, primPath, name, role);
                    case 3:
                        return printAttributeValue<true, double, 3>(fabricStage, primPath, name, role);
                    case 4:
                        return printAttributeValue<true, double, 4>(fabricStage, primPath, name, role);
                    case 6:
                        return printAttributeValue<true, double, 6>(fabricStage, primPath, name, role);
                    case 9:
                        return printAttributeValue<true, double, 9>(fabricStage, primPath, name, role);
                    case 16:
                        return printAttributeValue<true, double, 16>(fabricStage, primPath, name, role);
                    default:
                        break;
                }
                break;
            default:
                break;
        }
    }

    return std::string(TYPE_NOT_SUPPORTED_STRING);
}

} // namespace

std::string printFabricStage(omni::fabric::StageReaderWriter& fabricStage) {
    std::stringstream stream;

    // For extra debugging. This gets printed to the console.
    fabricStage.printBucketNames();

    // This returns ALL the buckets
    const auto& buckets = fabricStage.findPrims({});

    for (uint64_t bucketId = 0; bucketId < buckets.bucketCount(); ++bucketId) {
        const auto& attributes = fabricStage.getAttributeNamesAndTypes(buckets, bucketId);
        const auto& primPaths = fabricStage.getPathArray(buckets, bucketId);

        for (const auto& primPath : primPaths) {
            const auto primPathString = primPath.getString();
            const auto primPathUint64 = primPath.getHash();

            stream << fmt::format("Prim: {} ({})\n", primPathString, primPathUint64);
            stream << fmt::format("  Attributes:\n");

            for (const auto& attribute : attributes) {
                const auto attributeName = attribute.name.getText();
                const auto attributeType = attribute.type.getTypeName();
                const auto attributeBaseType = attribute.type.baseType;
                const auto attributeValue = printAttributeValue(fabricStage, primPath, attribute);

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

FabricStatistics getStatistics(omni::fabric::StageReaderWriter& fabricStage) {
    FabricStatistics statistics;

    const auto geometryBuckets = fabricStage.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId())},
        {omni::fabric::AttrNameAndType(FabricTypes::Mesh, FabricTokens::Mesh())});

    const auto materialBuckets = fabricStage.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId())},
        {omni::fabric::AttrNameAndType(FabricTypes::Material, FabricTokens::Material())});

    for (uint64_t bucketId = 0; bucketId < geometryBuckets.bucketCount(); ++bucketId) {
        const auto paths = fabricStage.getPathArray(geometryBuckets, bucketId);

        statistics.geometriesCapacity += paths.size();

        for (const auto& path : paths) {
            const auto worldVisibilityFabric = fabricStage.getAttributeRd<bool>(path, FabricTokens::_worldVisibility());
            const auto faceVertexCountsFabric =
                fabricStage.getArrayAttributeRd<int>(path, FabricTokens::faceVertexCounts());
            const auto tilesetIdFabric = fabricStage.getAttributeRd<int64_t>(path, FabricTokens::_cesium_tilesetId());

            assert(worldVisibilityFabric);
            assert(tilesetIdFabric);

            if (*tilesetIdFabric == NO_TILESET_ID) {
                continue;
            }

            ++statistics.geometriesLoaded;

            const auto triangleCount = faceVertexCountsFabric.size();
            statistics.trianglesLoaded += triangleCount;

            if (*worldVisibilityFabric) {
                ++statistics.geometriesRendered;
                statistics.trianglesRendered += triangleCount;
            }
        }
    }

    for (uint64_t bucketId = 0; bucketId < materialBuckets.bucketCount(); ++bucketId) {
        auto paths = fabricStage.getPathArray(materialBuckets, bucketId);

        const auto tilesetIdFabric =
            fabricStage.getAttributeArrayRd<int64_t>(materialBuckets, bucketId, FabricTokens::_cesium_tilesetId());

        statistics.materialsCapacity += paths.size();

        for (uint64_t i = 0; i < paths.size(); ++i) {
            if (tilesetIdFabric[i] == NO_TILESET_ID) {
                continue;
            }

            ++statistics.materialsLoaded;
        }
    }

    return statistics;
}

void destroyPrim(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    fabricStage.destroyPrim(path);

    // Prims removed from Fabric need special handling for their removal to be reflected in the Hydra render index
    // This workaround may not be needed in future Kit versions, but is needed as of Kit 105.0
    const auto changeTrackingPath = omni::fabric::Path::createImmortal("/TempChangeTracking");

    if (!fabricStage.getAttributeRd<omni::fabric::Path>(changeTrackingPath, FabricTokens::_deletedPrims())) {
        return;
    }

    const auto deletedPrimsSize = fabricStage.getArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims());
    fabricStage.setArrayAttributeSize(changeTrackingPath, FabricTokens::_deletedPrims(), deletedPrimsSize + 1);
    const auto deletedPrimsFabric =
        fabricStage.getArrayAttributeWr<omni::fabric::Path>(changeTrackingPath, FabricTokens::_deletedPrims());

    deletedPrimsFabric[deletedPrimsSize] = path;
}

void setTilesetTransform(
    omni::fabric::StageReaderWriter& fabricStage,
    int64_t tilesetId,
    const glm::dmat4& ecefToPrimWorldTransform) {
    const auto buckets = fabricStage.findPrims(
        {omni::fabric::AttrNameAndType(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId())},
        {omni::fabric::AttrNameAndType(
            FabricTypes::_cesium_gltfLocalToEcefTransform, FabricTokens::_cesium_gltfLocalToEcefTransform())});

    for (uint64_t bucketId = 0; bucketId < buckets.bucketCount(); ++bucketId) {
        // clang-format off
        const auto tilesetIdFabric = fabricStage.getAttributeArrayRd<int64_t>(buckets, bucketId, FabricTokens::_cesium_tilesetId());
        const auto gltfLocalToEcefTransformFabric = fabricStage.getAttributeArrayRd<pxr::GfMatrix4d>(buckets, bucketId, FabricTokens::_cesium_gltfLocalToEcefTransform());
        const auto extentFabric = fabricStage.getAttributeArrayRd<pxr::GfRange3d>(buckets, bucketId, FabricTokens::extent());
        const auto worldExtentFabric = fabricStage.getAttributeArrayWr<pxr::GfRange3d>(buckets, bucketId, FabricTokens::_worldExtent());
        const auto localMatrixFabric = fabricStage.getAttributeArrayWr<pxr::GfMatrix4d>(buckets, bucketId, FabricTokens::omni_fabric_localMatrix());
        const auto worldMatrixFabric = fabricStage.getAttributeArrayWr<pxr::GfMatrix4d>(buckets, bucketId, FabricTokens::omni_fabric_worldMatrix());
        // clang-format on

        for (uint64_t i = 0; i < tilesetIdFabric.size(); ++i) {
            if (tilesetIdFabric[i] == tilesetId) {
                const auto gltfLocalToEcefTransform = UsdUtil::usdToGlmMatrix(gltfLocalToEcefTransformFabric[i]);
                const auto gltfLocalToPrimWorldTransform = ecefToPrimWorldTransform * gltfLocalToEcefTransform;
                const auto gltfLocalExtent = UsdUtil::usdToGlmExtent(extentFabric[i]);
                const auto primWorldExtent = MathUtil::transformExtent(gltfLocalExtent, gltfLocalToPrimWorldTransform);
                localMatrixFabric[i] = UsdUtil::glmToUsdMatrix(gltfLocalToPrimWorldTransform);
                worldMatrixFabric[i] = UsdUtil::glmToUsdMatrix(gltfLocalToPrimWorldTransform);
                worldExtentFabric[i] = UsdUtil::glmToUsdExtent(primWorldExtent);
            }
        }
    }
}

omni::fabric::Path toFabricPath(const pxr::SdfPath& path) {
    return omni::fabric::getOrCreateImmortalPath(path);
}

omni::fabric::Token toFabricToken(const pxr::TfToken& token) {
    return omni::fabric::Token::createImmortal(token.GetText());
}

omni::fabric::Path joinPaths(const omni::fabric::Path& absolutePath, const omni::fabric::Token& relativePath) {
    return omni::fabric::Path::createImmortal(fmt::format("{}/{}", absolutePath.getString(), relativePath.getText()).c_str());
}

omni::fabric::Path getCopiedShaderPath(const omni::fabric::Path& materialPath, const omni::fabric::Path& shaderPath) {
    // materialPath is the FabricMaterial path
    // shaderPath is the USD shader path
    return FabricUtil::joinPaths(materialPath, omni::fabric::Token::createImmortal(UsdUtil::getSafeName(shaderPath.getString()).c_str()));
}

namespace {

struct FabricConnection {
    omni::fabric::Connection* pConnection;
    omni::fabric::Token attributeName;
};

std::vector<FabricConnection>
getConnections(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    std::vector<FabricConnection> connections;

    const auto attributes = fabricStage.getAttributeNamesAndTypes(path);
    const auto& names = attributes.first;
    const auto& types = attributes.second;

    for (uint64_t i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        const auto& type = types[i];
        if (type.baseType == omni::fabric::BaseDataType::eConnection) {
            const auto pConnection = fabricStage.getConnection(path, name);
            if (pConnection) {
                // In C++ 20 this can be emplace_back without the {}
                connections.push_back({pConnection, name});
            }
        }
    }

    return connections;
}

bool isOutput(const omni::fabric::Token& attributeName) {
    return attributeName == FabricTokens::outputs_out();
}

bool isConnection(const omni::fabric::Type& attributeType) {
    return attributeType.baseType == omni::fabric::BaseDataType::eConnection;
}

bool isEmptyToken(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const omni::fabric::Token& attributeName,
    const omni::fabric::Type& attributeType) {
    if (attributeType.baseType == omni::fabric::BaseDataType::eToken) {
        const auto pAttributeValue = fabricStage.getAttributeRd<omni::fabric::Token>(path, attributeName);
        if (!pAttributeValue || pAttributeValue->isNull()) {
            return true;
        }
    }

    return false;
}

std::vector<omni::fabric::Token>
getAttributesToCopy(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    std::vector<omni::fabric::Token> attributeNames;

    const auto attributes = fabricStage.getAttributeNamesAndTypes(path);
    const auto& names = attributes.first;
    const auto& types = attributes.second;

    for (uint64_t i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        const auto& type = types[i];

        if (!isOutput(name) && !isConnection(type) && !isEmptyToken(fabricStage, path, name, type)) {
            attributeNames.push_back(name);
        }
    }

    return attributeNames;
}

struct FabricAttribute {
    omni::fabric::Token name;
    omni::fabric::Type type;
};

std::vector<FabricAttribute>
getAttributesToCreate(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    std::vector<FabricAttribute> attributeNames;

    const auto attributes = fabricStage.getAttributeNamesAndTypes(path);
    const auto& names = attributes.first;
    const auto& types = attributes.second;

    for (uint64_t i = 0; i < names.size(); ++i) {
        const auto& name = names[i];
        const auto& type = types[i];

        if (isOutput(name) || isEmptyToken(fabricStage, path, name, type)) {
            // In C++ 20 this can be emplace_back without the {}
            attributeNames.push_back({name, type});
        }
    }

    return attributeNames;
}

void getConnectedPrimsRecursive(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    std::vector<omni::fabric::Path>& connectedPaths) {
    const auto connections = getConnections(fabricStage, path);
    for (const auto& connection : connections) {
        if (!CppUtil::contains(connectedPaths, connection.pConnection->path)) {
            connectedPaths.push_back(connection.pConnection->path);
            getConnectedPrimsRecursive(fabricStage, connection.pConnection->path, connectedPaths);
        }
    }
}

std::vector<omni::fabric::Path>
getPrimsInMaterialNetwork(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    std::vector<omni::fabric::Path> paths;
    paths.push_back(path);
    getConnectedPrimsRecursive(fabricStage, path, paths);
    return paths;
}

omni::fabric::Path getMaterialSource(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    if (fabricStage.attributeExistsWithType(path, FabricTokens::_materialSource(), FabricTypes::_materialSource)) {
        const auto materialSourceFabric =
            fabricStage.getArrayAttributeRd<omni::fabric::Path>(path, FabricTokens::_materialSource());
        if (!materialSourceFabric.empty()) {
            return *materialSourceFabric.begin();
        }
    }

    return path;
}

} // namespace

std::vector<omni::fabric::Path> copyMaterial(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& srcMaterialPath,
    const omni::fabric::Path& dstMaterialPath) {
    const auto iFabricStage = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();

    const auto materialSourcePath = getMaterialSource(fabricStage, srcMaterialPath);

    const auto srcPaths = getPrimsInMaterialNetwork(fabricStage, materialSourcePath);

    std::vector<omni::fabric::Path> dstPaths;
    dstPaths.reserve(srcPaths.size());

    for (const auto& srcPath : srcPaths) {
        auto dstPath = omni::fabric::Path();

        if (srcPath == materialSourcePath) {
            dstPath = dstMaterialPath;
        } else {
            const auto name = omni::fabric::Token::createImmortal(std::strrchr(srcPath.getString().c_str(), '/') + 1);
            dstPath = FabricUtil::getCopiedShaderPath(dstMaterialPath, srcMaterialPath.appendChild(fabricStage.getFabricId(), name));
        }

        dstPaths.push_back(dstPath);

        fabricStage.createPrim(dstPath);

        // This excludes connections, outputs, and empty tokens
        // The material network will be reconnected later once all the prims have been copied
        // The reason for excluding outputs and empty tokens is so that Omniverse doesn't print the warning
        //   [Warning] [omni.fabric.plugin] Warning: input has no valid data
        const auto attributesToCopy = getAttributesToCopy(fabricStage, srcPath);

        iFabricStage->copySpecifiedAttributes(
            fabricStage.getId(),
            srcPath,
            attributesToCopy.data(),
            dstPath,
            attributesToCopy.data(),
            attributesToCopy.size());

        // Add the outputs and empty tokens back. This doesn't print a warning.
        const auto attributesToCreate = getAttributesToCreate(fabricStage, srcPath);
        for (const auto& attribute : attributesToCreate) {
            fabricStage.createAttribute(dstPath, attribute.name, attribute.type);
        }
    }

    // Reconnect the prims
    for (uint64_t i = 0; i < srcPaths.size(); ++i) {
        const auto& srcPath = srcPaths[i];
        const auto& dstPath = dstPaths[i];
        const auto connections = getConnections(fabricStage, srcPath);
        for (const auto& connection : connections) {
            const auto index = CppUtil::indexOf(srcPaths, connection.pConnection->path);
            assert(index != srcPaths.size()); // Ensure that all connections are part of the material network
            const auto dstConnection =
                omni::fabric::Connection{dstPaths[index], connection.pConnection->attrName};
            fabricStage.createConnection(dstPath, connection.attributeName, dstConnection);
        }
    }

    return dstPaths;
}

bool materialHasCesiumNodes(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& materialPath) {
    const auto materialSourcePath = getMaterialSource(fabricStage, materialPath);

    const auto paths = getPrimsInMaterialNetwork(fabricStage, materialSourcePath);

    for (const auto& path : paths) {
        const auto mdlIdentifier = getMdlIdentifier(fabricStage, path);
        if (isCesiumNode(mdlIdentifier)) {
            return true;
        }
    }

    return false;
}

bool isCesiumNode(const omni::fabric::Token& mdlIdentifier) {
    return mdlIdentifier == FabricTokens::cesium_base_color_texture_float4() ||
           mdlIdentifier == FabricTokens::cesium_raster_overlay_float4() ||
           mdlIdentifier == FabricTokens::cesium_feature_id_int() || isCesiumPropertyNode(mdlIdentifier);
}

bool isCesiumPropertyNode(const omni::fabric::Token& mdlIdentifier) {
    return mdlIdentifier == FabricTokens::cesium_property_int() || mdlIdentifier == FabricTokens::cesium_property_int2() ||
           mdlIdentifier == FabricTokens::cesium_property_int3() || mdlIdentifier == FabricTokens::cesium_property_int4() ||
           mdlIdentifier == FabricTokens::cesium_property_float() ||
           mdlIdentifier == FabricTokens::cesium_property_float2() ||
           mdlIdentifier == FabricTokens::cesium_property_float3() ||
           mdlIdentifier == FabricTokens::cesium_property_float4();
}

bool isShaderConnectedToMaterial(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& materialPath,
    const omni::fabric::Path& shaderPath) {

    const auto materialSourcePath = getMaterialSource(fabricStage, materialPath);

    const auto paths = getPrimsInMaterialNetwork(fabricStage, materialSourcePath);
    return CppUtil::contains(paths, shaderPath);
}

omni::fabric::Token getMdlIdentifier(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path) {
    if (fabricStage.attributeExists(path, FabricTokens::info_mdl_sourceAsset_subIdentifier())) {
        const auto pInfoMdlSourceAssetSubIdentifierFabric =
            fabricStage.getAttributeRd<omni::fabric::Token>(path, FabricTokens::info_mdl_sourceAsset_subIdentifier());
        if (pInfoMdlSourceAssetSubIdentifierFabric) {
            return *pInfoMdlSourceAssetSubIdentifierFabric;
        }
    }
    return {};
}

omni::fabric::Type getPrimvarType(DataType type) {
    const auto baseDataType = DataTypeUtil::getPrimvarBaseDataType(type);
    const auto componentCount = DataTypeUtil::getComponentCount(type);
    return {baseDataType, static_cast<uint8_t>(componentCount), 1, omni::fabric::AttributeRole::eNone};
}

MdlExternalPropertyType getMdlExternalPropertyType(const omni::fabric::Token& mdlIdentifier) {
    assert(isCesiumPropertyNode(mdlIdentifier));

    if (mdlIdentifier == FabricTokens::cesium_property_int()) {
        return MdlExternalPropertyType::INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_int2()) {
        return MdlExternalPropertyType::VEC2_INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_int3()) {
        return MdlExternalPropertyType::VEC3_INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_int4()) {
        return MdlExternalPropertyType::VEC4_INT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float()) {
        return MdlExternalPropertyType::FLOAT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float2()) {
        return MdlExternalPropertyType::VEC2_FLOAT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float3()) {
        return MdlExternalPropertyType::VEC3_FLOAT32;
    } else if (mdlIdentifier == FabricTokens::cesium_property_float4()) {
        return MdlExternalPropertyType::VEC4_FLOAT32;
    }

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

    return false;
}

omni::fabric::Token getDynamicTextureProviderAssetPathToken(const std::string_view& name) {
    return omni::fabric::Token::createImmortal(fmt::format("dynamic://{}", name).c_str());
}

} // namespace cesium::omniverse::FabricUtil
