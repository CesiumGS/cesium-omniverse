#include "cesium/omniverse/UsdGeometry.h"

#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/OmniMaterial.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <glm/gtc/random.hpp>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>

namespace cesium::omniverse {

namespace {

const auto DEFAULT_VERTEX_COLOR = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_VERTEX_OPACITY = 1.0f;
const auto DEFAULT_EXTENT = pxr::VtArray<pxr::GfVec3f>{pxr::GfVec3f(0.0f, 0.0f, 0.0f), pxr::GfVec3f(0.0f, 0.0f, 0.0f)};
const auto DEFAULT_POSITION = pxr::GfVec3d(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = pxr::GfVec3f(0.0f, 0.0f, 0.0f);
const auto DEFAULT_SCALE = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_MATRIX = pxr::GfMatrix4d(1.0);

template <typename T> gsl::span<T> getSpan(pxr::VtArray<T>& array) {
    return gsl::span<T>(array.data(), array.size());
}

pxr::VtArray<pxr::GfVec3f> getExtent(const pxr::GfRange3d& range) {
    const auto& min = range.GetMin();
    const auto& max = range.GetMax();

    auto extent = pxr::VtArray<pxr::GfVec3f>(2);

    extent[0] = pxr::GfVec3f(static_cast<float>(min[0]), static_cast<float>(min[1]), static_cast<float>(min[2]));
    extent[1] = pxr::GfVec3f(static_cast<float>(max[0]), static_cast<float>(max[1]), static_cast<float>(max[2]));

    return extent;
}

pxr::GfVec3f getEulerRotation(const pxr::GfQuatf& quat) {
    const auto rotation = pxr::GfRotation(quat);
    const auto euler = rotation.Decompose(pxr::GfVec3d::XAxis(), pxr::GfVec3d::YAxis(), pxr::GfVec3d::ZAxis());
    return pxr::GfVec3f(static_cast<float>(euler[0]), static_cast<float>(euler[1]), static_cast<float>(euler[2]));
}

} // namespace

UsdGeometry::UsdGeometry(pxr::SdfPath path, const OmniGeometryDefinition& geometryDefinition, bool debugRandomColors)
    : OmniGeometry(path, geometryDefinition, debugRandomColors) {
    initialize();
}

UsdGeometry::~UsdGeometry() {
    const auto stage = UsdUtil::getUsdStage();
    stage->RemovePrim(_path);
}

void UsdGeometry::setVisibility(bool visible) {
    _mesh.GetPrim().SetActive(visible);
}

void UsdGeometry::assignMaterial(std::shared_ptr<OmniMaterial> material) {
    (void)material;
}

void UsdGeometry::initialize() {
    const auto hasMaterial = _geometryDefinition.hasMaterial();
    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto doubleSided = _geometryDefinition.getDoubleSided();

    (void)hasMaterial;

    const auto stage = UsdUtil::getUsdStage();
    auto mesh = pxr::UsdGeomMesh::Define(stage, _path);
    auto prim = mesh.GetPrim();

    mesh.CreateFaceVertexCountsAttr();
    mesh.CreateFaceVertexIndicesAttr();
    mesh.CreatePointsAttr();
    mesh.CreateExtentAttr();

    if (hasTexcoords) {
        mesh.CreatePrimvar(UsdTokens::primvars_st, pxr::SdfValueTypeNames->TexCoord2fArray, pxr::UsdGeomTokens->vertex);
    }

    if (hasNormals) {
        mesh.CreateNormalsAttr();
        mesh.SetNormalsInterpolation(UsdTokens::vertex);
    }

    if (hasVertexColors) {
        mesh.CreatePrimvar(
            UsdTokens::primvars_vertexColor, pxr::SdfValueTypeNames->Color3fArray, pxr::UsdGeomTokens->vertex);
    }

    mesh.GetDoubleSidedAttr().Set(doubleSided);
    mesh.GetSubdivisionSchemeAttr().Set(UsdTokens::none);

    mesh.CreateDisplayColorPrimvar(UsdTokens::constant);
    mesh.CreateDisplayOpacityPrimvar(UsdTokens::constant);

    prim.SetMetadata(UsdTokens::_cesium_tilesetId, int64_t(0));
    prim.SetMetadata(UsdTokens::_cesium_tileId, int64_t(0));
    prim.SetMetadata(UsdTokens::_cesium_localToEcefTransform, pxr::GfMatrix4d());

    reset();
}

void UsdGeometry::reset() {
    if (!UsdUtil::hasStage()) {
        return;
    }

    const auto hasTexcoords = _geometryDefinition.hasTexcoords();
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();

    const auto stage = UsdUtil::getUsdStage();
    auto mesh = _mesh;
    auto prim = mesh.GetPrim();

    mesh.GetExtentAttr().Set(DEFAULT_EXTENT);
    prim.SetActive(false);
    prim.SetMetadata(UsdTokens::_cesium_tilesetId, int64_t(-1));
    prim.SetMetadata(UsdTokens::_cesium_tileId, int64_t(-1));
    prim.SetMetadata(UsdTokens::_cesium_localToEcefTransform, DEFAULT_MATRIX);

    auto xformCommonApi = pxr::UsdGeomXformCommonAPI(prim);
    xformCommonApi.SetTranslate(DEFAULT_POSITION);
    xformCommonApi.SetRotate(DEFAULT_ORIENTATION);
    xformCommonApi.SetScale(DEFAULT_SCALE);

    mesh.GetFaceVertexCountsAttr().Set(pxr::VtArray<int>());
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtArray<int>());
    mesh.GetPointsAttr().Set(pxr::VtArray<pxr::GfVec3f>());

    if (hasTexcoords) {
        mesh.GetPrimvar(UsdTokens::primvars_st).Set(pxr::VtArray<pxr::GfVec2f>());
    }

    if (hasNormals) {
        mesh.GetNormalsAttr().Set(pxr::VtArray<pxr::GfVec3f>());
    }

    if (hasVertexColors) {
        mesh.GetPrimvar(UsdTokens::primvars_vertexColor).Set(pxr::VtArray<pxr::GfVec3f>());
    }
}

void UsdGeometry::setTile(
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

    auto mesh = _mesh;
    auto prim = mesh.GetPrim();

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
    const auto localExtent = GltfUtil::getExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

    if (positions.size() == 0 || indices.size() == 0 || !localExtent.has_value()) {
        return;
    }

    const auto localToEcefTransform = gltfToEcefTransform * nodeTransform;
    const auto localToUsdTransform = ecefToUsdTransform * localToEcefTransform;
    const auto [worldPosition, worldOrientation, worldScale] = UsdUtil::glmToUsdMatrixDecomposed(localToUsdTransform);

    auto faceVertexCountsUsd = pxr::VtArray<int>(faceVertexCounts.size());
    auto faceVertexCountsSpan = getSpan(faceVertexCountsUsd);
    faceVertexCounts.fill(faceVertexCountsSpan);

    auto indicesUsd = pxr::VtArray<int>(indices.size());
    auto indicesSpan = getSpan(indicesUsd);
    indices.fill(indicesSpan);

    auto positionsUsd = pxr::VtArray<pxr::GfVec3f>(positions.size());
    auto positionsSpan = getSpan(positionsUsd);
    positions.fill(positionsSpan);

    mesh.GetFaceVertexCountsAttr().Set(pxr::VtValue::Take(faceVertexCountsUsd));
    mesh.GetFaceVertexIndicesAttr().Set(pxr::VtValue::Take(indicesUsd));
    mesh.GetPointsAttr().Set(pxr::VtValue::Take(positionsUsd));

    mesh.GetExtentAttr().Set(getExtent(localExtent.value()));

    prim.SetMetadata(UsdTokens::_cesium_tilesetId, tilesetId);
    prim.SetMetadata(UsdTokens::_cesium_tileId, tileId);
    prim.SetMetadata(UsdTokens::_cesium_localToEcefTransform, UsdUtil::glmToUsdMatrix(localToEcefTransform));

    auto xformCommonApi = pxr::UsdGeomXformCommonAPI(prim);
    xformCommonApi.SetTranslate(worldPosition);
    xformCommonApi.SetRotate(getEulerRotation(worldOrientation));
    xformCommonApi.SetScale(worldScale);

    if (_debugRandomColors) {
        const auto r = glm::linearRand(0.0f, 1.0f);
        const auto g = glm::linearRand(0.0f, 1.0f);
        const auto b = glm::linearRand(0.0f, 1.0f);

        pxr::VtArray<pxr::GfVec3f> displayColor(1);
        displayColor[0] = pxr::GfVec3f(r, g, b);
        mesh.GetDisplayColorPrimvar().Set(displayColor);
    } else {
        pxr::VtArray<pxr::GfVec3f> displayColor(1);
        displayColor[0] = DEFAULT_VERTEX_COLOR;
        mesh.GetDisplayColorPrimvar().Set(displayColor);
    }

    mesh.GetDisplayOpacityPrimvar().Set(DEFAULT_VERTEX_OPACITY);

    if (hasTexcoords) {
        const auto& texcoords = hasImagery ? imageryTexcoords : texcoords_0;

        auto texcoordsUsd = pxr::VtArray<pxr::GfVec2f>(texcoords.size());
        auto texcoordsSpan = getSpan(texcoordsUsd);
        texcoords.fill(texcoordsSpan);

        mesh.GetPrimvar(UsdTokens::primvars_st).Set(pxr::VtValue::Take(texcoordsUsd));
    }

    if (hasNormals) {
        auto normalsUsd = pxr::VtArray<pxr::GfVec3f>(normals.size());
        auto normalsSpan = getSpan(normalsUsd);
        normals.fill(normalsSpan);

        mesh.GetNormalsAttr().Set(pxr::VtValue::Take(normalsUsd));
    }

    if (hasVertexColors) {
        auto vertexColorsUsd = pxr::VtArray<pxr::GfVec3f>(vertexColors.size());
        auto vertexColorSpan = getSpan(vertexColorsUsd);
        vertexColors.fill(vertexColorSpan);

        mesh.GetPrimvar(UsdTokens::primvars_vertexColor).Set(pxr::VtValue::Take(vertexColorsUsd));
    }
}

}; // namespace cesium::omniverse
