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
#include <omni/fabric/FabricUSD.h>
#include <pxr/base/gf/range3d.h>

namespace cesium::omniverse {

namespace {

const auto MATERIAL_LOADING_COLOR = pxr::GfVec3f(1.0f, 0.0f, 0.0f);
const auto DEFAULT_COLOR = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_EXTENT = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));
const auto DEFAULT_POSITION = pxr::GfVec3d(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = pxr::GfQuatf(1.0f, 0.0, 0.0, 0.0);
const auto DEFAULT_SCALE = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_MATRIX = pxr::GfMatrix4d(1.0);

} // namespace

FabricGeometry::FabricGeometry(pxr::SdfPath path, const FabricGeometryDefinition& geometryDefinition)
    : _path(path)
    , _geometryDefinition(geometryDefinition) {
    initialize();
}

FabricGeometry::~FabricGeometry() {
    FabricUtil::destroyPrim(_path);
}

void FabricGeometry::setActive(bool active) {
    if (!active) {
        reset();
    }
}

void FabricGeometry::setVisibility(bool visible) {
    auto sip = UsdUtil::getFabricStageReaderWriter();

    auto worldVisibilityFabric =
        sip.getAttributeWr<bool>(omni::fabric::asInt(_path), FabricTokens::_worldVisibility);
    *worldVisibilityFabric = visible;
}

pxr::SdfPath FabricGeometry::getPath() const {
    return _path;
}

const FabricGeometryDefinition& FabricGeometry::getGeometryDefinition() const {
    return _geometryDefinition;
}

void FabricGeometry::assignMaterial(std::shared_ptr<FabricMaterial> material) {
    if (_geometryDefinition.hasMaterial())
    {
        auto sip = UsdUtil::getFabricStageReaderWriter();
        auto materialIdFabric =
            sip.getArrayAttributeWr<omni::fabric::Path>(omni::fabric::asInt(_path), FabricTokens::materialId);
        if (materialIdFabric.size() == 1)
        {
            materialIdFabric[0] = omni::fabric::asInt(material->getPath());
        }
    }
}

void FabricGeometry::initialize() {
    const auto hasMaterial = _geometryDefinition.hasMaterial();
    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto doubleSided = _geometryDefinition.getDoubleSided();

    auto sip = UsdUtil::getFabricStageReaderWriter();
    const auto pathFabric = omni::fabric::Path(omni::fabric::asInt(_path));

    sip.createPrim(pathFabric);

    FabricAttributesBuilder attributes;
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::_localExtent, FabricTokens::_localExtent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::primvars, FabricTokens::primvars);
    attributes.addAttribute(FabricTypes::primvarInterpolations, FabricTokens::primvarInterpolations);
    attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
    attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
    attributes.addAttribute(FabricTypes::_cesium_localToEcefTransform, FabricTokens::_cesium_localToEcefTransform);
    attributes.addAttribute(FabricTypes::_worldPosition, FabricTokens::_worldPosition);
    attributes.addAttribute(FabricTypes::_worldOrientation, FabricTokens::_worldOrientation);
    attributes.addAttribute(FabricTypes::_worldScale, FabricTokens::_worldScale);
    attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided);
    attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme);

    attributes.addAttribute(FabricTypes::materialId, FabricTokens::materialId);

    if (hasTexcoords) {
        attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st);
    }

    if (hasNormals) {
        attributes.addAttribute(FabricTypes::primvars_normals, FabricTokens::primvars_normals);
    }

    attributes.createAttributes(pathFabric);

    // clang-format off
    auto doubleSidedFabric = sip.getAttributeWr<bool>(pathFabric, FabricTokens::doubleSided);
    auto subdivisionSchemeFabric = sip.getAttributeWr<omni::fabric::Token>(pathFabric, FabricTokens::subdivisionScheme);
    // clang-format on

    *subdivisionSchemeFabric = FabricTokens::none;
    *doubleSidedFabric = doubleSided;

    // Initialize primvars
    size_t primvarsCount = 0;
    size_t primvarIndexSt = 0;
    size_t primvarIndexNormal = 0;

    const size_t primvarIndexDisplayColor = primvarsCount++;

    if (hasTexcoords) {
        primvarIndexSt = primvarsCount++;
    }

    if (hasNormals) {
        primvarIndexNormal = primvarsCount++;
    }

    sip.setArrayAttributeSize(pathFabric, FabricTokens::primvars, primvarsCount);
    sip.setArrayAttributeSize(pathFabric, FabricTokens::primvarInterpolations, primvarsCount);
    sip.setArrayAttributeSize(pathFabric, FabricTokens::materialId, hasMaterial ? 1 : 0);

    // clang-format off
    auto primvarsFabric = sip.getArrayAttributeWr<omni::fabric::Token>(pathFabric, FabricTokens::primvars);
    auto primvarInterpolationsFabric = sip.getArrayAttributeWr<omni::fabric::Token>(pathFabric, FabricTokens::primvarInterpolations);
    // clang-format on

    primvarsFabric[primvarIndexDisplayColor] = FabricTokens::primvars_displayColor;
    primvarInterpolationsFabric[primvarIndexDisplayColor] = FabricTokens::constant;

    if (hasTexcoords) {
        primvarsFabric[primvarIndexSt] = FabricTokens::primvars_st;
        primvarInterpolationsFabric[primvarIndexSt] = FabricTokens::vertex;
    }

    if (hasNormals) {
        primvarsFabric[primvarIndexNormal] = FabricTokens::primvars_normals;
        primvarInterpolationsFabric[primvarIndexNormal] = FabricTokens::vertex;
    }

    reset();
}

void FabricGeometry::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();

    auto sip = UsdUtil::getFabricStageReaderWriter();
    const auto pathFabric = omni::fabric::Path(omni::fabric::asInt(_path));

    auto localExtentFabric = sip.getAttributeWr<pxr::GfRange3d>(pathFabric, FabricTokens::_localExtent);
    auto worldExtentFabric = sip.getAttributeWr<pxr::GfRange3d>(pathFabric, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = sip.getAttributeWr<bool>(pathFabric, FabricTokens::_worldVisibility);
    auto tilesetIdFabric = sip.getAttributeWr<int64_t>(pathFabric, FabricTokens::_cesium_tilesetId);
    auto tileIdFabric = sip.getAttributeWr<int64_t>(pathFabric, FabricTokens::_cesium_tileId);
    auto localToEcefTransformFabric =
        sip.getAttributeWr<pxr::GfMatrix4d>(pathFabric, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = sip.getAttributeWr<pxr::GfVec3d>(pathFabric, FabricTokens::_worldPosition);
    auto worldOrientationFabric = sip.getAttributeWr<pxr::GfQuatf>(pathFabric, FabricTokens::_worldOrientation);
    auto worldScaleFabric = sip.getAttributeWr<pxr::GfVec3f>(pathFabric, FabricTokens::_worldScale);

    *localExtentFabric = DEFAULT_EXTENT;
    *worldExtentFabric = DEFAULT_EXTENT;
    *worldVisibilityFabric = false;
    *tilesetIdFabric = -1;
    *tileIdFabric = -1;
    *localToEcefTransformFabric = DEFAULT_MATRIX;
    *worldPositionFabric = DEFAULT_POSITION;
    *worldOrientationFabric = DEFAULT_ORIENTATION;
    *worldScaleFabric = DEFAULT_SCALE;

    sip.setArrayAttributeSize(pathFabric, FabricTokens::faceVertexCounts, 0);
    sip.setArrayAttributeSize(pathFabric, FabricTokens::faceVertexIndices, 0);
    sip.setArrayAttributeSize(pathFabric, FabricTokens::points, 0);

    if (hasTexcoords) {
        sip.setArrayAttributeSize(pathFabric, FabricTokens::primvars_st, 0);
    }

    if (hasNormals) {
        sip.setArrayAttributeSize(pathFabric, FabricTokens::primvars_normals, 0);
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

    const auto hasMaterial = _geometryDefinition.hasMaterial();
    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();

    auto sip = UsdUtil::getFabricStageReaderWriter();
    const auto pathFabric = omni::fabric::Path(omni::fabric::asInt(_path));

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto indices = GltfUtil::getIndices(model, primitive, positions);
    const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
    const auto texcoords_0 = GltfUtil::getTexcoords(model, primitive, 0, glm::fvec2(0.0, 0.0), glm::fvec2(1.0, 1.0));
    const auto imageryTexcoords = GltfUtil::getImageryTexcoords(
        model,
        primitive,
        imageryTexcoordSetIndex,
        glm::fvec2(imageryTexcoordTranslation),
        glm::fvec2(imageryTexcoordScale));
    const auto localExtent = GltfUtil::getExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

    if (positions.empty() || indices.empty() || !localExtent.has_value()) {
        return;
    }

    const auto localToEcefTransform = gltfToEcefTransform * nodeTransform;
    const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
    const auto [worldPosition, worldOrientation, worldScale] = UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);
    const auto worldExtent = UsdUtil::computeWorldExtent(localExtent.value(), localToUsdTransform);

    sip.setArrayAttributeSize(pathFabric, FabricTokens::faceVertexCounts, faceVertexCounts.size());
    sip.setArrayAttributeSize(pathFabric, FabricTokens::faceVertexIndices, indices.size());
    sip.setArrayAttributeSize(pathFabric, FabricTokens::points, positions.size());
    sip.setArrayAttributeSize(pathFabric, FabricTokens::primvars_displayColor, 1);

    // clang-format off
    auto faceVertexCountsFabric = sip.getArrayAttributeWr<int>(pathFabric, FabricTokens::faceVertexCounts);
    auto faceVertexIndicesFabric = sip.getArrayAttributeWr<int>(pathFabric, FabricTokens::faceVertexIndices);
    auto pointsFabric = sip.getArrayAttributeWr<pxr::GfVec3f>(pathFabric, FabricTokens::points);
    auto localExtentFabric = sip.getAttributeWr<pxr::GfRange3d>(pathFabric, FabricTokens::_localExtent);
    auto worldExtentFabric = sip.getAttributeWr<pxr::GfRange3d>(pathFabric, FabricTokens::_worldExtent);
    auto tilesetIdFabric = sip.getAttributeWr<int64_t>(pathFabric, FabricTokens::_cesium_tilesetId);
    auto tileIdFabric = sip.getAttributeWr<int64_t>(pathFabric, FabricTokens::_cesium_tileId);
    auto localToEcefTransformFabric = sip.getAttributeWr<pxr::GfMatrix4d>(pathFabric, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = sip.getAttributeWr<pxr::GfVec3d>(pathFabric, FabricTokens::_worldPosition);
    auto worldOrientationFabric = sip.getAttributeWr<pxr::GfQuatf>(pathFabric, FabricTokens::_worldOrientation);
    auto worldScaleFabric = sip.getAttributeWr<pxr::GfVec3f>(pathFabric, FabricTokens::_worldScale);
    auto displayColorFabric = sip.getArrayAttributeWr<pxr::GfVec3f>(pathFabric, FabricTokens::primvars_displayColor);
    // clang-format on

    std::copy(faceVertexCounts.begin(), faceVertexCounts.end(), faceVertexCountsFabric.begin());
    std::copy(indices.begin(), indices.end(), faceVertexIndicesFabric.begin());
    std::copy(positions.begin(), positions.end(), pointsFabric.begin());

    *localExtentFabric = localExtent.value();
    *worldExtentFabric = worldExtent;
    *tilesetIdFabric = tilesetId;
    *tileIdFabric = tileId;
    *localToEcefTransformFabric = UsdUtil::glmToUsdMatrix(localToEcefTransform);
    *worldPositionFabric = worldPosition;
    *worldOrientationFabric = worldOrientation;
    *worldScaleFabric = worldScale;

    displayColorFabric[0] = hasMaterial ? MATERIAL_LOADING_COLOR : DEFAULT_COLOR;

    if (hasTexcoords) {
        const auto& texcoords = hasImagery ? imageryTexcoords : texcoords_0;

        sip.setArrayAttributeSize(pathFabric, FabricTokens::primvars_st, texcoords.size());

        auto stFabric = sip.getArrayAttributeWr<pxr::GfVec2f>(pathFabric, FabricTokens::primvars_st);

        std::copy(texcoords.begin(), texcoords.end(), stFabric.begin());
    }

    if (hasNormals) {
        sip.setArrayAttributeSize(pathFabric, FabricTokens::primvars_normals, normals.size());

        auto normalsFabric = sip.getArrayAttributeWr<pxr::GfVec3f>(pathFabric, FabricTokens::primvars_normals);

        std::copy(normals.begin(), normals.end(), normalsFabric.begin());
    }
}

}; // namespace cesium::omniverse
