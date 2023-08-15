#include "cesium/omniverse/FabricGeometry.h"

#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <glm/gtc/random.hpp>
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

} // namespace

FabricGeometry::FabricGeometry(
    const pxr::SdfPath& path,
    const FabricGeometryDefinition& geometryDefinition,
    bool debugRandomColors,
    long stageId)
    : _pathFabric(path.GetText())
    , _geometryDefinition(geometryDefinition)
    , _debugRandomColors(debugRandomColors)
    , _stageId(stageId) {
    if (stageDestroyed()) {
        return;
    }

    initialize();
    reset();
}

FabricGeometry::~FabricGeometry() {
    if (stageDestroyed()) {
        return;
    }

    FabricUtil::destroyPrim(_pathFabric);
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

    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_pathFabric, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = visible;
}

omni::fabric::Path FabricGeometry::getPathFabric() const {
    return _pathFabric;
}

const FabricGeometryDefinition& FabricGeometry::getGeometryDefinition() const {
    return _geometryDefinition;
}

void FabricGeometry::setMaterial(const omni::fabric::Path& materialPath) {
    if (stageDestroyed()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::material_binding, 1);
    auto materialBindingFabric = srw.getArrayAttributeWr<uint64_t>(_pathFabric, FabricTokens::material_binding);
    materialBindingFabric[0] = omni::fabric::PathC(materialPath).path;
}

void FabricGeometry::initialize() {
    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto doubleSided = _geometryDefinition.getDoubleSided();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    srw.createPrim(_pathFabric);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars, FabricTokens::primvars);
    attributes.addAttribute(FabricTypes::primvarInterpolations, FabricTokens::primvarInterpolations);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::primvars_displayOpacity, FabricTokens::primvars_displayOpacity);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
    attributes.addAttribute(FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided);
    attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme);
    attributes.addAttribute(FabricTypes::material_binding, FabricTokens::material_binding);

    if (hasTexcoords) {
        attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st);
    }

    if (hasNormals) {
        attributes.addAttribute(FabricTypes::primvars_normals, FabricTokens::primvars_normals);
    }

    if (hasVertexColors) {
        attributes.addAttribute(FabricTypes::primvars_vertexColor, FabricTokens::primvars_vertexColor);
    }

    attributes.createAttributes(_pathFabric);

    // clang-format off
    auto doubleSidedFabric = srw.getAttributeWr<bool>(_pathFabric, FabricTokens::doubleSided);
    auto subdivisionSchemeFabric = srw.getAttributeWr<omni::fabric::Token>(_pathFabric, FabricTokens::subdivisionScheme);
    // clang-format on

    *doubleSidedFabric = doubleSided;
    *subdivisionSchemeFabric = FabricTokens::none;

    // Initialize primvars
    size_t primvarsCount = 0;
    size_t primvarIndexSt = 0;
    size_t primvarIndexNormal = 0;
    size_t primvarIndexVertexColor = 0;

    const size_t primvarIndexDisplayColor = primvarsCount++;
    const size_t primvarIndexDisplayOpacity = primvarsCount++;

    if (hasTexcoords) {
        primvarIndexSt = primvarsCount++;
    }

    if (hasNormals) {
        primvarIndexNormal = primvarsCount++;
    }

    if (hasVertexColors) {
        primvarIndexVertexColor = primvarsCount++;
    }

    srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars, primvarsCount);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvarInterpolations, primvarsCount);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_displayColor, 1);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_displayOpacity, 1);

    // clang-format off
    auto primvarsFabric = srw.getArrayAttributeWr<omni::fabric::Token>(_pathFabric, FabricTokens::primvars);
    auto primvarInterpolationsFabric = srw.getArrayAttributeWr<omni::fabric::Token>(_pathFabric, FabricTokens::primvarInterpolations);
    // clang-format on

    primvarsFabric[primvarIndexDisplayColor] = FabricTokens::primvars_displayColor;
    primvarsFabric[primvarIndexDisplayOpacity] = FabricTokens::primvars_displayOpacity;

    primvarInterpolationsFabric[primvarIndexDisplayColor] = FabricTokens::constant;
    primvarInterpolationsFabric[primvarIndexDisplayOpacity] = FabricTokens::constant;

    if (hasTexcoords) {
        primvarsFabric[primvarIndexSt] = FabricTokens::primvars_st;
        primvarInterpolationsFabric[primvarIndexSt] = FabricTokens::vertex;
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
    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    // clang-format off
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_pathFabric, FabricTokens::_worldVisibility);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_pathFabric, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_pathFabric, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_pathFabric, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::_worldScale);
    auto displayColorFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::primvars_displayColor);
    auto displayOpacityFabric = srw.getArrayAttributeWr<float>(_pathFabric, FabricTokens::primvars_displayOpacity);
    // clang-format on

    *extentFabric = DEFAULT_EXTENT;
    *worldExtentFabric = DEFAULT_EXTENT;
    *worldVisibilityFabric = DEFAULT_VISIBILITY;
    *localToEcefTransformFabric = DEFAULT_MATRIX;
    *worldPositionFabric = DEFAULT_POSITION;
    *worldOrientationFabric = DEFAULT_ORIENTATION;
    *worldScaleFabric = DEFAULT_SCALE;
    displayColorFabric[0] = DEFAULT_VERTEX_COLOR;
    displayOpacityFabric[0] = DEFAULT_VERTEX_OPACITY;

    FabricUtil::setTilesetId(_pathFabric, NO_TILESET_ID);

    srw.setArrayAttributeSize(_pathFabric, FabricTokens::material_binding, 0);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexCounts, 0);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexIndices, 0);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::points, 0);

    if (hasTexcoords) {
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_st, 0);
    }

    if (hasNormals) {
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_normals, 0);
    }

    if (hasVertexColors) {
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_vertexColor, 0);
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
    bool hasImagery) {

    if (stageDestroyed()) {
        return;
    }

    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto indices = GltfUtil::getIndices(model, primitive, positions);
    const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
    const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);
    const auto texcoords_0 = GltfUtil::getTexcoords(model, primitive, 0);
    const auto imageryTexcoords = GltfUtil::getImageryTexcoords(model, primitive, 0);
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

    if (primitive.mode == 0) { //if tile is a point cloud

        auto numVoxels = positions.size();
        const float quadHalfSize = 1.5f;
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::points, static_cast<size_t>(numVoxels * 8));
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexCounts, numVoxels * 2 * 6);
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_displayColor, numVoxels * 8);

        srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexIndices, numVoxels * 6 * 2 * 3);

        auto pointsFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::points);
        auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_pathFabric, FabricTokens::faceVertexCounts);
        auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_pathFabric, FabricTokens::faceVertexIndices);

        std::vector<glm::fvec3> vertexColorsData(numVoxels);
        gsl::span<glm::fvec3> vertexColorsSpan(vertexColorsData);
        if (hasVertexColors) {
            vertexColors.fill(vertexColorsSpan);
            srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_vertexColor, numVoxels * 8);
        }
        auto vertexColorsFabric = srw.getArrayAttributeWr<glm::fvec3>(_pathFabric, FabricTokens::primvars_vertexColor);

        size_t vertIndex = 0;
        size_t vertexCountsIndex = 0;
        size_t faceVertexIndex = 0;
        size_t voxelCounter = 0;
        size_t vertexColorsIndex = 0;
        for (size_t voxelNum = 0; voxelNum < numVoxels; voxelNum++) {
            auto centerGlm = positions.get(voxelNum);
            pxr::GfVec3f center{centerGlm.x, centerGlm.y, centerGlm.z};

            pointsFabric[vertIndex++] = pxr::GfVec3f{-quadHalfSize, -quadHalfSize, -quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{-quadHalfSize, quadHalfSize, -quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{quadHalfSize, quadHalfSize, -quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{quadHalfSize, -quadHalfSize, -quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{-quadHalfSize, -quadHalfSize, quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{-quadHalfSize, quadHalfSize, quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{quadHalfSize, quadHalfSize, quadHalfSize} + center;
            pointsFabric[vertIndex++] = pxr::GfVec3f{quadHalfSize, -quadHalfSize, quadHalfSize} + center;

            for (int i = 0; i < 6; i++) {
                faceVertexCountsFabric[vertexCountsIndex++] = 3;
                faceVertexCountsFabric[vertexCountsIndex++] = 3;
            }

            //front
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelCounter * 8);
            //left
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelCounter * 8);
            //right
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelCounter * 8);
            //top
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(voxelCounter * 8);
            //bottom
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(voxelCounter * 8);
            //back
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(voxelCounter * 8);
            faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(voxelCounter * 8);

            voxelCounter++;

            auto color = vertexColorsSpan[voxelNum];
            if (hasVertexColors) {
                for (int i = 0; i < 8; i++) {
                    vertexColorsFabric[vertexColorsIndex++] = color;
                }
            }
        }

        // clang-format off
        auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::extent);
        auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::_worldExtent);
        auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_pathFabric, FabricTokens::_cesium_localToEcefTransform);
        auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_pathFabric, FabricTokens::_worldPosition);
        auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_pathFabric, FabricTokens::_worldOrientation);
        auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::_worldScale);
        // clang-format on

        *extentFabric = localExtent;
        *worldExtentFabric = worldExtent;
        *localToEcefTransformFabric = UsdUtil::glmToUsdMatrix(localToEcefTransform);
        *worldPositionFabric = worldPosition;
        *worldOrientationFabric = worldOrientation;
        *worldScaleFabric = worldScale;
    } else {
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexCounts, faceVertexCounts.size());
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexIndices, indices.size());
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::points, positions.size());

        // clang-format off
        auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_pathFabric, FabricTokens::faceVertexCounts);
        auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_pathFabric, FabricTokens::faceVertexIndices);
        auto pointsFabric = srw.getArrayAttributeWr<glm::fvec3>(_pathFabric, FabricTokens::points);
        auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::extent);
        auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::_worldExtent);
        auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_pathFabric, FabricTokens::_cesium_localToEcefTransform);
        auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_pathFabric, FabricTokens::_worldPosition);
        auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_pathFabric, FabricTokens::_worldOrientation);
        auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::_worldScale);
        auto displayColorFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::primvars_displayColor);
        auto displayOpacityFabric = srw.getArrayAttributeWr<float>(_pathFabric, FabricTokens::primvars_displayOpacity);
        // clang-format on

        faceVertexCounts.fill(faceVertexCountsFabric);
        indices.fill(faceVertexIndicesFabric);
        positions.fill(pointsFabric);

        *extentFabric = localExtent;
        *worldExtentFabric = worldExtent;
        *localToEcefTransformFabric = UsdUtil::glmToUsdMatrix(localToEcefTransform);
        *worldPositionFabric = worldPosition;
        *worldOrientationFabric = worldOrientation;
        *worldScaleFabric = worldScale;

        FabricUtil::setTilesetId(_pathFabric, tilesetId);

        if (_debugRandomColors) {
            const auto r = glm::linearRand(0.0f, 1.0f);
            const auto g = glm::linearRand(0.0f, 1.0f);
            const auto b = glm::linearRand(0.0f, 1.0f);
            displayColorFabric[0] = pxr::GfVec3f(r, g, b);
        } else {
            displayColorFabric[0] = DEFAULT_VERTEX_COLOR;
        }

        displayOpacityFabric[0] = DEFAULT_VERTEX_OPACITY;

        if (hasTexcoords) {
            const auto& texcoords = hasImagery ? imageryTexcoords : texcoords_0;

            srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_st, texcoords.size());

            auto stFabric = srw.getArrayAttributeWr<glm::fvec2>(_pathFabric, FabricTokens::primvars_st);

            texcoords.fill(stFabric);
        }

        if (hasNormals) {
            srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_normals, normals.size());

            auto normalsFabric = srw.getArrayAttributeWr<glm::fvec3>(_pathFabric, FabricTokens::primvars_normals);

            normals.fill(normalsFabric);
        }

        if (hasVertexColors) {
            srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_vertexColor, vertexColors.size());

            auto vertexColorsFabric =
                srw.getArrayAttributeWr<glm::fvec3>(_pathFabric, FabricTokens::primvars_vertexColor);

            vertexColors.fill(vertexColorsFabric);
        }
    }
}

bool FabricGeometry::stageDestroyed() {
    // Add this guard to all public member functions, including constructors and destructors. Tile render resources can
    // continue to be processed asynchronously even after the tileset and USD stage have been destroyed, so prevent any
    // operations that would modify the stage.
    return _stageId != UsdUtil::getUsdStageId();
}

}; // namespace cesium::omniverse
