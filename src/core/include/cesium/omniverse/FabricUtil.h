#pragma once

#include <glm/fwd.hpp>
#include <pxr/usd/usd/common.h>

#include <string>

namespace omni::fabric {
class StageReaderWriter;
class Path;
class Token;
struct Type;
} // namespace omni::fabric

namespace cesium::omniverse {

enum class DataType;
enum class MdlExternalPropertyType;
enum class MdlInternalPropertyType;
struct FabricStatistics;

} // namespace cesium::omniverse

namespace cesium::omniverse::FabricUtil {

// -1 means the prim is not associated with a tileset yet
const int64_t NO_TILESET_ID{-1};

std::string printFabricStage(omni::fabric::StageReaderWriter& fabricStage);
FabricStatistics getStatistics(omni::fabric::StageReaderWriter& fabricStage);
void destroyPrim(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path);
void setTilesetTransform(
    omni::fabric::StageReaderWriter& fabricStage,
    int64_t tilesetId,
    const glm::dmat4& ecefToPrimWorldTransform);
omni::fabric::Path toFabricPath(const PXR_NS::SdfPath& path);
omni::fabric::Token toFabricToken(const PXR_NS::TfToken& token);
omni::fabric::Path joinPaths(const omni::fabric::Path& absolutePath, const omni::fabric::Token& relativePath);
omni::fabric::Path getCopiedShaderPath(const omni::fabric::Path& materialPath, const omni::fabric::Path& shaderPath);
std::vector<omni::fabric::Path> copyMaterial(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& srcMaterialPath,
    const omni::fabric::Path& dstMaterialPath);
bool materialHasCesiumNodes(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path);
bool isCesiumNode(const omni::fabric::Token& mdlIdentifier);
bool isCesiumPropertyNode(const omni::fabric::Token& mdlIdentifier);
bool isShaderConnectedToMaterial(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& materialPath,
    const omni::fabric::Path& shaderPath);
omni::fabric::Token getMdlIdentifier(omni::fabric::StageReaderWriter& fabricStage, const omni::fabric::Path& path);
omni::fabric::Type getPrimvarType(DataType type);
MdlExternalPropertyType getMdlExternalPropertyType(const omni::fabric::Token& mdlIdentifier);
bool typesCompatible(MdlExternalPropertyType externalType, MdlInternalPropertyType internalType);

} // namespace cesium::omniverse::FabricUtil
