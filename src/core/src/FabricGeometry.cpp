#include "cesium/omniverse/FabricGeometry.h"

#include "cesium/omniverse/DataType.h"
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

const auto DEFAULT_DOUBLE_SIDED = false;
const auto DEFAULT_EXTENT = pxr::GfRange3d(pxr::GfVec3d(0.0, 0.0, 0.0), pxr::GfVec3d(0.0, 0.0, 0.0));
const auto DEFAULT_POSITION = pxr::GfVec3d(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = pxr::GfQuatf(1.0f, 0.0, 0.0, 0.0);
const auto DEFAULT_SCALE = pxr::GfVec3f(1.0f, 1.0f, 1.0f);
const auto DEFAULT_MATRIX = pxr::GfMatrix4d(1.0);
const auto DEFAULT_VISIBILITY = false;

template <DataType T>
void setVertexAttributeValues(
    omni::fabric::StageReaderWriter& srw,
    const omni::fabric::Path& path,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const VertexAttributeInfo& attribute,
    uint64_t repeat) {

    const auto accessor = GltfUtil::getVertexAttributeValues<T>(model, primitive, attribute.gltfAttributeName);
    assert(accessor.size() > 0);
    srw.setArrayAttributeSize(path, attribute.fabricAttributeName, accessor.size() * repeat);
    auto fabricValues = srw.getArrayAttributeWr<GetPrimvarType<T>>(path, attribute.fabricAttributeName);
    accessor.fill(fabricValues, repeat);
}

void setVertexAttributeValues(
    omni::fabric::StageReaderWriter& srw,
    const omni::fabric::Path& path,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const VertexAttributeInfo& attribute,
    uint64_t repeat) {

    assert(isPrimvarType(attribute.type));

    // clang-format off
    switch (attribute.type) {
        case DataType::UINT8: setVertexAttributeValues<DataType::UINT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::INT8: setVertexAttributeValues<DataType::INT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::UINT16: setVertexAttributeValues<DataType::UINT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::INT16: setVertexAttributeValues<DataType::INT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::FLOAT32: setVertexAttributeValues<DataType::FLOAT32>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::UINT8_NORM: setVertexAttributeValues<DataType::UINT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::INT8_NORM: setVertexAttributeValues<DataType::INT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::UINT16_NORM: setVertexAttributeValues<DataType::UINT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::INT16_NORM: setVertexAttributeValues<DataType::INT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_UINT8: setVertexAttributeValues<DataType::VEC2_UINT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_INT8: setVertexAttributeValues<DataType::VEC2_INT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_UINT16: setVertexAttributeValues<DataType::VEC2_UINT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_INT16: setVertexAttributeValues<DataType::VEC2_INT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_FLOAT32: setVertexAttributeValues<DataType::VEC2_FLOAT32>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_UINT8_NORM: setVertexAttributeValues<DataType::VEC2_UINT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_INT8_NORM: setVertexAttributeValues<DataType::VEC2_INT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_UINT16_NORM: setVertexAttributeValues<DataType::VEC2_UINT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC2_INT16_NORM: setVertexAttributeValues<DataType::VEC2_INT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_UINT8: setVertexAttributeValues<DataType::VEC3_UINT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_INT8: setVertexAttributeValues<DataType::VEC3_INT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_UINT16: setVertexAttributeValues<DataType::VEC3_UINT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_INT16: setVertexAttributeValues<DataType::VEC3_INT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_FLOAT32: setVertexAttributeValues<DataType::VEC3_FLOAT32>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_UINT8_NORM: setVertexAttributeValues<DataType::VEC3_UINT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_INT8_NORM: setVertexAttributeValues<DataType::VEC3_INT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_UINT16_NORM: setVertexAttributeValues<DataType::VEC3_UINT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC3_INT16_NORM: setVertexAttributeValues<DataType::VEC3_INT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_UINT8: setVertexAttributeValues<DataType::VEC4_UINT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_INT8: setVertexAttributeValues<DataType::VEC4_INT8>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_UINT16: setVertexAttributeValues<DataType::VEC4_UINT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_INT16: setVertexAttributeValues<DataType::VEC4_INT16>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_FLOAT32: setVertexAttributeValues<DataType::VEC4_FLOAT32>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_UINT8_NORM: setVertexAttributeValues<DataType::VEC4_UINT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_INT8_NORM: setVertexAttributeValues<DataType::VEC4_INT8_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_UINT16_NORM: setVertexAttributeValues<DataType::VEC4_UINT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        case DataType::VEC4_INT16_NORM: setVertexAttributeValues<DataType::VEC4_INT16_NORM>(srw, path, model, primitive, attribute, repeat); break;
        default:
            // Not a valid vertex attribute type
            assert(false);
    }
    // clang-format on
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
    const auto texcoordSetCount = _geometryDefinition.getTexcoordSetCount();
    const auto& customVertexAttributes = _geometryDefinition.getCustomVertexAttributes();
    const auto customVertexAttributesCount = customVertexAttributes.size();
    const auto hasVertexIds = _geometryDefinition.hasVertexIds();

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
        attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st_n(i));
    }

    if (hasNormals) {
        attributes.addAttribute(FabricTypes::primvars_normals, FabricTokens::primvars_normals);
    }

    if (hasVertexColors) {
        attributes.addAttribute(FabricTypes::primvars_COLOR_0, FabricTokens::primvars_COLOR_0);
    }

    if (hasVertexIds) {
        attributes.addAttribute(FabricTypes::primvars_vertexId, FabricTokens::primvars_vertexId);
    }

    for (const auto& customVertexAttribute : customVertexAttributes) {
        attributes.addAttribute(
            getFabricPrimvarType(customVertexAttribute.type), customVertexAttribute.fabricAttributeName);
    }

    attributes.createAttributes(_path);

    auto subdivisionSchemeFabric = srw.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::subdivisionScheme);
    *subdivisionSchemeFabric = FabricTokens::none;

    // Initialize primvars
    size_t primvarsCount = 0;
    size_t primvarIndexNormal = 0;
    size_t primvarIndexVertexColor = 0;
    size_t primvarIndexVertexId = 0;

    std::vector<uint64_t> primvarIndexStArray;
    primvarIndexStArray.reserve(texcoordSetCount);

    for (uint64_t i = 0; i < texcoordSetCount; i++) {
        primvarIndexStArray.push_back(primvarsCount++);
    }

    std::vector<uint64_t> primvarIndexCustomVertexAttributesArray;
    primvarIndexCustomVertexAttributesArray.reserve(customVertexAttributesCount);

    for (uint64_t i = 0; i < customVertexAttributesCount; i++) {
        primvarIndexCustomVertexAttributesArray.push_back(primvarsCount++);
    }

    if (hasNormals) {
        primvarIndexNormal = primvarsCount++;
    }

    if (hasVertexColors) {
        primvarIndexVertexColor = primvarsCount++;
    }

    if (hasVertexIds) {
        primvarIndexVertexId = primvarsCount++;
    }

    srw.setArrayAttributeSize(_path, FabricTokens::primvars, primvarsCount);
    srw.setArrayAttributeSize(_path, FabricTokens::primvarInterpolations, primvarsCount);

    // clang-format off
    auto primvarsFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvars);
    auto primvarInterpolationsFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvarInterpolations);
    // clang-format on

    for (uint64_t i = 0; i < texcoordSetCount; i++) {
        primvarsFabric[primvarIndexStArray[i]] = FabricTokens::primvars_st_n(i);
        primvarInterpolationsFabric[primvarIndexStArray[i]] = FabricTokens::vertex;
    }

    for (uint64_t i = 0; i < customVertexAttributesCount; i++) {
        const auto& customVertexAttribute = *std::next(customVertexAttributes.begin(), static_cast<int>(i));
        primvarsFabric[primvarIndexCustomVertexAttributesArray[i]] = customVertexAttribute.fabricAttributeName;
        primvarInterpolationsFabric[primvarIndexCustomVertexAttributesArray[i]] = FabricTokens::vertex;
    }

    if (hasNormals) {
        primvarsFabric[primvarIndexNormal] = FabricTokens::primvars_normals;
        primvarInterpolationsFabric[primvarIndexNormal] = FabricTokens::vertex;
    }

    if (hasVertexColors) {
        primvarsFabric[primvarIndexVertexColor] = FabricTokens::primvars_COLOR_0;
        primvarInterpolationsFabric[primvarIndexVertexColor] = FabricTokens::vertex;
    }

    if (hasVertexIds) {
        primvarsFabric[primvarIndexVertexId] = FabricTokens::primvars_vertexId;
        primvarInterpolationsFabric[primvarIndexVertexId] = FabricTokens::vertex;
    }
}

void FabricGeometry::reset() {
    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto texcoordSetCount = _geometryDefinition.getTexcoordSetCount();
    const auto& customVertexAttributes = _geometryDefinition.getCustomVertexAttributes();
    const auto hasVertexIds = _geometryDefinition.hasVertexIds();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    // clang-format off
    auto doubleSidedFabric = srw.getAttributeWr<bool>(_path, FabricTokens::doubleSided);
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto worldVisibilityFabric = srw.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    // clang-format on

    *doubleSidedFabric = DEFAULT_DOUBLE_SIDED;
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
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_st_n(i), 0);
    }

    for (const auto& customVertexAttribute : customVertexAttributes) {
        srw.setArrayAttributeSize(_path, customVertexAttribute.fabricAttributeName, 0);
    }

    if (hasNormals) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_normals, 0);
    }

    if (hasVertexColors) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0, 0);
    }

    if (hasVertexIds) {
        srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId, 0);
    }
}

void FabricGeometry::setGeometry(
    int64_t tilesetId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& gltfToEcefTransform,
    const glm::dmat4& nodeTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const MaterialInfo& materialInfo,
    bool smoothNormals,
    const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
    const std::unordered_map<uint64_t, uint64_t>& imageryTexcoordIndexMapping) {

    if (stageDestroyed()) {
        return;
    }

    const auto hasNormals = _geometryDefinition.hasNormals();
    const auto hasVertexColors = _geometryDefinition.hasVertexColors();
    const auto& customVertexAttributes = _geometryDefinition.getCustomVertexAttributes();
    const auto hasVertexIds = _geometryDefinition.hasVertexIds();

    auto srw = UsdUtil::getFabricStageReaderWriter();

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto indices = GltfUtil::getIndices(model, primitive, positions);
    const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
    const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);
    const auto vertexIds = GltfUtil::getVertexIds(positions);
    const auto extent = GltfUtil::getExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

    if (positions.size() == 0 || indices.size() == 0 || !extent.has_value()) {
        return;
    }

    const auto doubleSided = materialInfo.doubleSided;
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

        if (hasVertexColors) {
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0, numVoxels * 8);
            auto vertexColorsFabric = srw.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::primvars_COLOR_0);
            vertexColors.fill(vertexColorsFabric, 8);
        }

        if (hasVertexIds) {
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId, numVoxels * 8);
            auto vertexIdsFabric = srw.getArrayAttributeWr<float>(_path, FabricTokens::primvars_vertexId);
            vertexIds.fill(vertexIdsFabric, 8);
        }

        for (const auto& customVertexAttribute : customVertexAttributes) {
            setVertexAttributeValues(srw, _path, model, primitive, customVertexAttribute, 8);
        }

        size_t vertIndex = 0;
        size_t vertexCountsIndex = 0;
        size_t faceVertexIndex = 0;
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
            const auto& primvarStToken = FabricTokens::primvars_st_n(texcoordIndex);
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
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0, vertexColors.size());

            auto vertexColorsFabric = srw.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::primvars_COLOR_0);

            vertexColors.fill(vertexColorsFabric);
        }

        if (hasVertexIds) {
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId, vertexIds.size());

            auto vertexIdsFabric = srw.getArrayAttributeWr<float>(_path, FabricTokens::primvars_vertexId);

            vertexIds.fill(vertexIdsFabric);
        }

        for (const auto& customVertexAttribute : customVertexAttributes) {
            setVertexAttributeValues(srw, _path, model, primitive, customVertexAttribute, 1);
        }
    }

    // clang-format off
    auto doubleSidedFabric = srw.getAttributeWr<bool>(_path, FabricTokens::doubleSided);
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    // clang-format on

    *doubleSidedFabric = doubleSided;
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
