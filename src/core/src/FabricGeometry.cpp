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
    const omni::fabric::Path& path,
    const FabricGeometryDefinition& geometryDefinition,
    bool debugRandomColors,
    long stageId)
    : _path(path)
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

omni::fabric::Path FabricGeometry::getPath() const {
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
    auto materialBindingFabric = srw.getArrayAttributeWr<uint64_t>(_path, FabricTokens::material_binding);
    materialBindingFabric[0] = omni::fabric::PathC(materialPath).path;
}

void FabricGeometry::initialize() {
    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto doubleSided = _geometryDefinition.getDoubleSided();

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

    attributes.createAttributes(_path);

    // clang-format off
    auto doubleSidedFabric = srw.getAttributeWr<bool>(_path, FabricTokens::doubleSided);
    auto subdivisionSchemeFabric = srw.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::subdivisionScheme);
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

    srw.setArrayAttributeSize(_path, FabricTokens::primvars, primvarsCount);
    srw.setArrayAttributeSize(_path, FabricTokens::primvarInterpolations, primvarsCount);
    srw.setArrayAttributeSize(_path, FabricTokens::primvars_displayColor, 1);
    srw.setArrayAttributeSize(_path, FabricTokens::primvars_displayOpacity, 1);

    // clang-format off
    auto primvarsFabric = srw.getArrayAttributeWr<omni::fabric::Token>(_path, FabricTokens::primvars);
    auto primvarInterpolationsFabric = srw.getArrayAttributeWr<omni::fabric::Token>(_path, FabricTokens::primvarInterpolations);
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
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    auto displayColorFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_path, FabricTokens::primvars_displayColor);
    auto displayOpacityFabric = srw.getArrayAttributeWr<float>(_path, FabricTokens::primvars_displayOpacity);
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

    FabricUtil::setTilesetId(_path, NO_TILESET_ID);

    srw.setArrayAttributeSize(_path, FabricTokens::material_binding, 0);
    srw.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, 0);
    srw.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, 0);
    srw.setArrayAttributeSize(_path, FabricTokens::points, 0);

    if (hasTexcoords) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_st, 0);
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

    srw.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, faceVertexCounts.size());
    srw.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, indices.size());
    srw.setArrayAttributeSize(_path, FabricTokens::points, positions.size());

    // clang-format off
    auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts);
    auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices);
    auto pointsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points);
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    auto displayColorFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_path, FabricTokens::primvars_displayColor);
    auto displayOpacityFabric = srw.getArrayAttributeWr<float>(_path, FabricTokens::primvars_displayOpacity);
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

    FabricUtil::setTilesetId(_path, tilesetId);

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

        srw.setArrayAttributeSize(_path, FabricTokens::primvars_st, texcoords.size());

        auto stFabric = srw.getArrayAttributeWr<glm::fvec2>(_path, FabricTokens::primvars_st);

        texcoords.fill(stFabric);
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

bool FabricGeometry::stageDestroyed() {
    // Add this guard to all public member functions, including constructors and destructors. Tile render resources can
    // continue to be processed asynchronously even after the tileset and USD stage have been destroyed, so prevent any
    // operations that would modify the stage.
    return _stageId != UsdUtil::getUsdStageId();
}

}; // namespace cesium::omniverse
