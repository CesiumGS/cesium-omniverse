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
#include <pxr/base/gf/range3d.h>

namespace cesium::omniverse {

namespace {

const auto DEFAULT_VERTEX_COLOR = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_VERTEX_OPACITY = 1.0f;
const auto DEFAULT_EXTENT = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));
const auto DEFAULT_POSITION = pxr::GfVec3d(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = pxr::GfQuatf(1.0f, 0.0, 0.0, 0.0);
const auto DEFAULT_SCALE = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_MATRIX = pxr::GfMatrix4d(1.0);

} // namespace

FabricGeometry::FabricGeometry(
    const pxr::SdfPath& path,
    const FabricGeometryDefinition& geometryDefinition,
    bool debugRandomColors)
    : _pathFabric(path.GetText())
    , _geometryDefinition(geometryDefinition)
    , _debugRandomColors(debugRandomColors) {
    initialize();
}

FabricGeometry::~FabricGeometry() {
    FabricUtil::destroyPrim(_pathFabric);
}

void FabricGeometry::setActive(bool active) {
    if (!active) {
        reset();
    }
}

void FabricGeometry::setVisibility(bool visible) {
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

void FabricGeometry::assignMaterial(const std::shared_ptr<FabricMaterial>& material) {
    if (!_geometryDefinition.hasMaterial()) {
        return;
    }

    auto srw = UsdUtil::getFabricStageReaderWriter();
    auto materialBindingFabric = srw.getArrayAttributeWr<uint64_t>(_pathFabric, FabricTokens::materialBinding);
    if (materialBindingFabric.size() == 1) {
        materialBindingFabric[0] = omni::fabric::PathC(material->getPathFabric()).path;
    }
}

void FabricGeometry::initialize() {
    const auto hasMaterial = _geometryDefinition.hasMaterial();
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
    attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
    attributes.addAttribute(FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided);
    attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme);
    attributes.addAttribute(FabricTypes::materialBinding, FabricTokens::materialBinding);

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

    srw.setArrayAttributeSize(_pathFabric, FabricTokens::materialBinding, hasMaterial ? 1 : 0);

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

    reset();
}

void FabricGeometry::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    // clang-format off
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_pathFabric, FabricTokens::_worldVisibility);
    auto tilesetIdFabric = srw.getAttributeWr<int64_t>(_pathFabric, FabricTokens::_cesium_tilesetId);
    auto tileIdFabric = srw.getAttributeWr<int64_t>(_pathFabric, FabricTokens::_cesium_tileId);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_pathFabric, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_pathFabric, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_pathFabric, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::_worldScale);
    // clang-format on

    *extentFabric = DEFAULT_EXTENT;
    *worldExtentFabric = DEFAULT_EXTENT;
    *worldVisibilityFabric = false;
    *tilesetIdFabric = -1;
    *tileIdFabric = -1;
    *localToEcefTransformFabric = DEFAULT_MATRIX;
    *worldPositionFabric = DEFAULT_POSITION;
    *worldOrientationFabric = DEFAULT_ORIENTATION;
    *worldScaleFabric = DEFAULT_SCALE;

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

void FabricGeometry::setTile(
    int64_t tilesetId,
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& gltfToEcefTransform,
    const glm::dmat4& nodeTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    bool hasImagery,
    const glm::dvec2& imageryTexcoordTranslation,
    const glm::dvec2& imageryTexcoordScale,
    uint64_t imageryTexcoordSetIndex) {

    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto indices = GltfUtil::getIndices(model, primitive, positions);
    const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
    const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);
    const auto texcoords_0 = GltfUtil::getTexcoords(model, primitive, 0, glm::fvec2(0.0, 0.0), glm::fvec2(1.0, 1.0));
    const auto imageryTexcoords = GltfUtil::getImageryTexcoords(
        model,
        primitive,
        imageryTexcoordSetIndex,
        glm::fvec2(imageryTexcoordTranslation),
        glm::fvec2(imageryTexcoordScale));
    const auto extent = GltfUtil::getExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

    if (positions.size() == 0 || indices.size() == 0 || !extent.has_value()) {
        return;
    }

    const auto localToEcefTransform = gltfToEcefTransform * nodeTransform;
    const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
    const auto [worldPosition, worldOrientation, worldScale] = UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);
    const auto worldExtent = UsdUtil::computeWorldExtent(extent.value(), localToUsdTransform);

    srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexCounts, faceVertexCounts.size());
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::faceVertexIndices, indices.size());
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::points, positions.size());
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_displayColor, 1);
    srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_displayOpacity, 1);

    // clang-format off
    auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_pathFabric, FabricTokens::faceVertexCounts);
    auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_pathFabric, FabricTokens::faceVertexIndices);
    auto pointsFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::points);
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_pathFabric, FabricTokens::_worldExtent);
    auto tilesetIdFabric = srw.getAttributeWr<int64_t>(_pathFabric, FabricTokens::_cesium_tilesetId);
    auto tileIdFabric = srw.getAttributeWr<int64_t>(_pathFabric, FabricTokens::_cesium_tileId);
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

    *extentFabric = extent.value();
    *worldExtentFabric = worldExtent;
    *tilesetIdFabric = tilesetId;
    *tileIdFabric = tileId;
    *localToEcefTransformFabric = UsdUtil::glmToUsdMatrix(localToEcefTransform);
    *worldPositionFabric = worldPosition;
    *worldOrientationFabric = worldOrientation;
    *worldScaleFabric = worldScale;

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

        auto stFabric = srw.getArrayAttributeWr<pxr::GfVec2f>(_pathFabric, FabricTokens::primvars_st);

        texcoords.fill(stFabric);
    }

    if (hasNormals) {
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_normals, normals.size());

        auto normalsFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::primvars_normals);

        normals.fill(normalsFabric);
    }

    if (hasVertexColors) {
        srw.setArrayAttributeSize(_pathFabric, FabricTokens::primvars_vertexColor, vertexColors.size());

        auto vertexColorsFabric =
            srw.getArrayAttributeWr<pxr::GfVec3f>(_pathFabric, FabricTokens::primvars_vertexColor);

        vertexColors.fill(vertexColorsFabric);
    }
}

}; // namespace cesium::omniverse
