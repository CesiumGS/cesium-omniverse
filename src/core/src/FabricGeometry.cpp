#include "cesium/omniverse/FabricGeometry.h"

#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/fwd.hpp>

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <omni/fabric/FabricUSD.h>

namespace cesium::omniverse {

namespace {

const auto DEFAULT_VERTEX_COLOR = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_VERTEX_OPACITY = 1.0f;
const auto DEFAULT_EXTENT = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));
const auto DEFAULT_POSITION = pxr::GfVec3d(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = pxr::GfQuatf(1.0f, 0.0, 0.0, 0.0);
const auto DEFAULT_SCALE = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_MATRIX = pxr::GfMatrix4d(1.0);
const auto DEFAULT_VISIBILITY = false;

uint64_t getTexcoordSetCount(const FabricGeometryDefinition& geometryDefinition) {
    auto texcoordSetCount = geometryDefinition.getTexcoordSetCount();

    if (texcoordSetCount > FabricTokens::MAX_PRIMVAR_ST_COUNT) {
        CESIUM_LOG_WARN(
            "Number of texcoord sets ({}) exceeds maximum number of texcoord sets ({}). Textures using excess texcoord "
            "sets will be ignored.",
            texcoordSetCount,
            FabricTokens::MAX_PRIMVAR_ST_COUNT);
    }

    texcoordSetCount = glm::min(texcoordSetCount, FabricTokens::MAX_PRIMVAR_ST_COUNT);

    return texcoordSetCount;
}

} // namespace

FabricGeometry::FabricGeometry(
    const omni::fabric::Path& path,
    const FabricGeometryDefinition& geometryDefinition,
    long stageId)
    : _path(path)
    , _geometryDefinition(geometryDefinition)
    , _stageId(stageId) {
    if (stageDestroyed()) {
        return;
    }

    FabricResourceManager::getInstance().retainPath(path);

    initialize();
    reset();
}

FabricGeometry::~FabricGeometry() {
    if (stageDestroyed()) {
        return;
    }

    FabricUtil::destroyPrim(_path);
}

void FabricGeometry::setActive(bool active) {
    if (stageDestroyed()) {
        return;
    }

    if (!active) {
        reset();
    }
}

void FabricGeometry::setVisibility(bool visible) {
    if (stageDestroyed()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();

    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = visible;
}

const omni::fabric::Path& FabricGeometry::getPath() const {
    return _path;
}

const FabricGeometryDefinition& FabricGeometry::getGeometryDefinition() const {
    return _geometryDefinition;
}

void FabricGeometry::setMaterial(const omni::fabric::Path& materialPath) {
    if (stageDestroyed()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();
    srw.setArrayAttributeSize(_path, FabricTokens::material_binding, 1);
    auto materialBindingFabric = srw.getArrayAttributeWr<omni::fabric::PathC>(_path, FabricTokens::material_binding);
    materialBindingFabric[0] = materialPath;
}

void FabricGeometry::initialize() {
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto doubleSided = _geometryDefinition.getDoubleSided();
    const auto texcoordSetCount = getTexcoordSetCount(_geometryDefinition);

    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(_path);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars, FabricTokens::primvars);
    attributes.addAttribute(FabricTypes::primvarInterpolations, FabricTokens::primvarInterpolations);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
    attributes.addAttribute(FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided);
    attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme);
    attributes.addAttribute(FabricTypes::material_binding, FabricTokens::material_binding);

    for (uint64_t i = 0; i < texcoordSetCount; i++) {
        attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st_n[i]);
    }

    if (hasNormals) {
        attributes.addAttribute(FabricTypes::primvars_normals, FabricTokens::primvars_normals);
    }

    if (hasVertexColors) {
        attributes.addAttribute(FabricTypes::primvars_vertexColor, FabricTokens::primvars_vertexColor);
    }

    attributes.createAttributes(_path);

    // clang-format off
    auto doubleSidedFabric = srw.getAttributeWr<bool>(_path, FabricTokens::doubleSided);
    auto subdivisionSchemeFabric = srw.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::subdivisionScheme);
    // clang-format on

    *doubleSidedFabric = doubleSided;
    *subdivisionSchemeFabric = FabricTokens::none;

    // Initialize primvars
    size_t primvarsCount = 0;
    size_t primvarIndexNormal = 0;
    size_t primvarIndexVertexColor = 0;

    std::vector<uint64_t> primvarIndexStArray;
    primvarIndexStArray.reserve(texcoordSetCount);

    for (uint64_t i = 0; i < texcoordSetCount; i++) {
        primvarIndexStArray.push_back(primvarsCount++);
    }

    if (hasNormals) {
        primvarIndexNormal = primvarsCount++;
    }

    if (hasVertexColors) {
        primvarIndexVertexColor = primvarsCount++;
    }

    srw.setArrayAttributeSize(_path, FabricTokens::primvars, primvarsCount);
    srw.setArrayAttributeSize(_path, FabricTokens::primvarInterpolations, primvarsCount);

    // clang-format off
    auto primvarsFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvars);
    auto primvarInterpolationsFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvarInterpolations);
    // clang-format on

    for (uint64_t i = 0; i < texcoordSetCount; i++) {
        primvarsFabric[primvarIndexStArray[i]] = FabricTokens::primvars_st_n[i];
        primvarInterpolationsFabric[primvarIndexStArray[i]] = FabricTokens::vertex;
    }

    if (hasNormals) {
        primvarsFabric[primvarIndexNormal] = FabricTokens::primvars_normals;
        primvarInterpolationsFabric[primvarIndexNormal] = FabricTokens::vertex;
    }

    if (hasVertexColors) {
        primvarsFabric[primvarIndexVertexColor] = FabricTokens::primvars_vertexColor;
        primvarInterpolationsFabric[primvarIndexVertexColor] = FabricTokens::vertex;
    }
}

void FabricGeometry::reset() {
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto texcoordSetCount = getTexcoordSetCount(_geometryDefinition);

    auto srw = UsdUtil::getFabricStageReaderWriter();

    // clang-format off
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    // clang-format on

    *extentFabric = DEFAULT_EXTENT;
    *worldExtentFabric = DEFAULT_EXTENT;
    *worldVisibilityFabric = DEFAULT_VISIBILITY;
    *localToEcefTransformFabric = DEFAULT_MATRIX;
    *worldPositionFabric = DEFAULT_POSITION;
    *worldOrientationFabric = DEFAULT_ORIENTATION;
    *worldScaleFabric = DEFAULT_SCALE;

    FabricUtil::setTilesetId(_path, NO_TILESET_ID);

    srw.setArrayAttributeSize(_path, FabricTokens::material_binding, 0);
    srw.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, 0);
    srw.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, 0);
    srw.setArrayAttributeSize(_path, FabricTokens::points, 0);

    for (uint64_t i = 0; i < texcoordSetCount; i++) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_st_n[i], 0);
    }

    if (hasNormals) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_normals, 0);
    }

    if (hasVertexColors) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexColor, 0);
    }
}

void FabricGeometry::setGeometry(
    int64_t tilesetId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& gltfToEcefTransform,
    const glm::dmat4& nodeTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
    const std::unordered_map<uint64_t, uint64_t>& imageryTexcoordIndexMapping) {

    if (stageDestroyed()) {
        return;
    }

    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto indices = GltfUtil::getIndices(model, primitive, positions);
    const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
    const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);
    const auto extent = GltfUtil::getExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

    if (positions.size() == 0 || indices.size() == 0 || !extent.has_value()) {
        return;
    }

    const auto localExtent = UsdUtil::glmToUsdRange(extent.value());
    const auto localToEcefTransform = gltfToEcefTransform * nodeTransform;
    const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
    const auto [worldPosition, worldOrientation, worldScale] = UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);
    const auto worldExtent = UsdUtil::computeWorldExtent(localExtent, localToUsdTransform);

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::POINTS) {
        const auto numVoxels = positions.size();
        const auto shapeHalfSize = 1.5f;
        srw.setArrayAttributeSize(_path, FabricTokens::points, numVoxels * 8);
        srw.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, numVoxels * 2 * 6);
        srw.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, numVoxels * 6 * 2 * 3);

        auto pointsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points);
        auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts);
        auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices);

        std::vector<glm::fvec3> vertexColorsData(numVoxels);
        gsl::span<glm::fvec3> vertexColorsSpan(vertexColorsData);
        if (hasVertexColors) {
            vertexColors.fill(vertexColorsSpan);
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexColor, numVoxels * 8);
        }
        auto vertexColorsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::primvars_vertexColor);

        size_t vertIndex = 0;
        size_t vertexCountsIndex = 0;
        size_t faceVertexIndex = 0;
        size_t vertexColorsIndex = 0;
        for (size_t voxelIndex = 0; voxelIndex < numVoxels; voxelIndex++) {
            const auto& center = positions.get(voxelIndex);

            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, -shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, -shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, -shapeHalfSize, shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, shapeHalfSize, shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, shapeHalfSize, shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, -shapeHalfSize, shapeHalfSize} + center;

            for (int i = 0; i < 6; i++) {
                faceVertexCountsFabric[vertexCountsIndex++] = 3;
                faceVertexCountsFabric[vertexCountsIndex++] = 3;
            }

            // front
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelIndex * 8);
            // left
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelIndex * 8);
            // right
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelIndex * 8);
            // top
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelIndex * 8);
            // bottom
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelIndex * 8);
            // back
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelIndex * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelIndex * 8);

            if (hasVertexColors) {
                const auto& color = vertexColorsSpan[voxelIndex];
                for (int i = 0; i < 8; i++) {
                    vertexColorsFabric[vertexColorsIndex++] = color;
                }
            }
        }
    } else {
        srw.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, faceVertexCounts.size());
        srw.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, indices.size());
        srw.setArrayAttributeSize(_path, FabricTokens::points, positions.size());

        auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts);
        auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices);
        auto pointsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points);

        faceVertexCounts.fill(faceVertexCountsFabric);
        indices.fill(faceVertexIndicesFabric);
        positions.fill(pointsFabric);

        const auto fillTexcoords = [this, &srw](uint64_t texcoordIndex, const TexcoordsAccessor& texcoords) {
            assert(texcoordIndex < _geometryDefinition.getTexcoordSetCount());

            if (texcoordIndex >= FabricTokens::MAX_PRIMVAR_ST_COUNT) {
                return;
            }

            const auto& primvarStToken = FabricTokens::primvars_st_n[texcoordIndex];
            srw.setArrayAttributeSize(_path, primvarStToken, texcoords.size());
            auto stFabric = srw.getArrayAttributeWr<glm::fvec2>(_path, primvarStToken);
            texcoords.fill(stFabric);
        };

        for (const auto& [gltfSetIndex, primvarStIndex] : texcoordIndexMapping) {
            const auto texcoords = GltfUtil::getTexcoords(model, primitive, gltfSetIndex);
            fillTexcoords(primvarStIndex, texcoords);
        }

        for (const auto& [gltfSetIndex, primvarStIndex] : imageryTexcoordIndexMapping) {
            const auto texcoords = GltfUtil::getImageryTexcoords(model, primitive, gltfSetIndex);
            fillTexcoords(primvarStIndex, texcoords);
        }

        if (hasNormals) {
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_normals, normals.size());

            auto normalsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::primvars_normals);

            normals.fill(normalsFabric);
        }

        if (hasVertexColors) {
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexColor, vertexColors.size());

            auto vertexColorsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::primvars_vertexColor);

            vertexColors.fill(vertexColorsFabric);
        }
    }

    // clang-format off
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    // clang-format on

    *extentFabric = localExtent;
    *worldExtentFabric = worldExtent;
    *localToEcefTransformFabric = UsdUtil::glmToUsdMatrix(localToEcefTransform);
    *worldPositionFabric = worldPosition;
    *worldOrientationFabric = worldOrientation;
    *worldScaleFabric = worldScale;

    FabricUtil::setTilesetId(_path, tilesetId);
}

bool FabricGeometry::stageDestroyed() {
    // Add this guard to all public member functions, including constructors and destructors. Tile render resources can
    // continue to be processed asynchronously even after the tileset and USD stage have been destroyed, so prevent any
    // operations that would modify the stage.
    return _stageId != UsdUtil::getUsdStageId();
}

}; // namespace cesium::omniverse
