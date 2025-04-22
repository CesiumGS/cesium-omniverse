#include "cesium/omniverse/FabricGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/DataType.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/FabricVertexAttributeDescriptor.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/MathUtil.h"
#include "cesium/omniverse/UsdTokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/fwd.hpp>

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/SimStageWithHistory.h>
#include <pxr/base/gf/range3d.h>

namespace cesium::omniverse {

namespace {

const auto DEFAULT_DOUBLE_SIDED = false;
const auto DEFAULT_EXTENT = std::array<glm::dvec3, 2>{{glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 0.0)}};
const auto DEFAULT_POSITION = glm::dvec3(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = glm::dquat(1.0, 0.0, 0.0, 0.0);
const auto DEFAULT_SCALE = glm::dvec3(1.0, 1.0, 1.0);
const auto DEFAULT_MATRIX = glm::dmat4(1.0);
const auto DEFAULT_VISIBILITY = false;

template <DataType T>
void setVertexAttributeValues(
    omni::fabric::StageReaderWriter& fabricStage,
    const omni::fabric::Path& path,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricVertexAttributeDescriptor& attribute,
    uint64_t repeat) {

    const auto accessor = GltfUtil::getVertexAttributeValues<T>(model, primitive, attribute.gltfAttributeName);
    fabricStage.setArrayAttributeSize(path, attribute.fabricAttributeName, accessor.size() * repeat);
    const auto fabricValues =
        fabricStage.getArrayAttributeWr<DataTypeUtil::GetNativeType<DataTypeUtil::getPrimvarType<T>()>>(
            path, attribute.fabricAttributeName);
    accessor.fill(fabricValues, repeat);
}

} // namespace

FabricGeometry::FabricGeometry(
    Context* pContext,
    const omni::fabric::Path& path,
    const FabricGeometryDescriptor& geometryDescriptor,
    int64_t poolId)
    : _pContext(pContext)
    , _path(path)
    , _geometryDescriptor(geometryDescriptor)
    , _poolId(poolId)
    , _stageId(pContext->getUsdStageId()) {
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

    FabricUtil::destroyPrim(_pContext->getFabricStage(), _path);
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

    auto& fabricStage = _pContext->getFabricStage();

    const auto worldVisibilityFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility);
    *worldVisibilityFabric = visible;
}

const omni::fabric::Path& FabricGeometry::getPath() const {
    return _path;
}

const FabricGeometryDescriptor& FabricGeometry::getGeometryDescriptor() const {
    return _geometryDescriptor;
}

int64_t FabricGeometry::getPoolId() const {
    return _poolId;
}

void FabricGeometry::setMaterial(const omni::fabric::Path& materialPath) {
    if (stageDestroyed()) {
        return;
    }

    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.setArrayAttributeSize(_path, FabricTokens::material_binding, 1);
    const auto materialBindingFabric =
        fabricStage.getArrayAttributeWr<omni::fabric::PathC>(_path, FabricTokens::material_binding);
    materialBindingFabric[0] = materialPath;
}

void FabricGeometry::initialize() {
    const auto hasNormals = _geometryDescriptor.hasNormals();
    const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
    const auto texcoordSetCount = _geometryDescriptor.getTexcoordSetCount();
    const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
    const auto customVertexAttributesCount = customVertexAttributes.size();
    const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

    auto& fabricStage = _pContext->getFabricStage();

    fabricStage.createPrim(_path);

    // clang-format off
    FabricAttributesBuilder attributes(_pContext);
    attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts);
    attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices);
    attributes.addAttribute(FabricTypes::points, FabricTokens::points);
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent);
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent);
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility);
    attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh);
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId);
    attributes.addAttribute(FabricTypes::_cesium_gltfLocalToEcefTransform, FabricTokens::_cesium_gltfLocalToEcefTransform);
    attributes.addAttribute(FabricTypes::omni_fabric_localMatrix, FabricTokens::omni_fabric_localMatrix);
    attributes.addAttribute(FabricTypes::omni_fabric_worldMatrix, FabricTokens::omni_fabric_worldMatrix);
    attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided);
    attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme);
    attributes.addAttribute(FabricTypes::material_binding, FabricTokens::material_binding);
    // clang-format on

    for (uint64_t i = 0; i < texcoordSetCount; ++i) {
        attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st_n(i));
        attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::primvars_st_interpolation_n(i));
    }

    if (hasNormals) {
        attributes.addAttribute(FabricTypes::normals, FabricTokens::normals);
        attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::normals_interpolation);
    }

    if (hasVertexColors) {
        attributes.addAttribute(FabricTypes::primvars_COLOR_0, FabricTokens::primvars_COLOR_0);
        attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::primvars_COLOR_0_interpolation);
    }

    if (hasVertexIds) {
        attributes.addAttribute(FabricTypes::primvars_vertexId, FabricTokens::primvars_vertexId);
        attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::primvars_vertexId_interpolation);
    }

    for (const auto& customVertexAttribute : customVertexAttributes) {
        attributes.addAttribute(
            FabricUtil::getPrimvarType(customVertexAttribute.type), customVertexAttribute.fabricAttributeName);
        attributes.addAttribute(FabricTypes::primvars_interpolation, customVertexAttribute.fabricInterpolationName);
    }

    attributes.createAttributes(_path);

    const auto subdivisionSchemeFabric =
        fabricStage.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::subdivisionScheme);
    *subdivisionSchemeFabric = FabricTokens::none;

    const auto localMatrixFabric =
        fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::omni_fabric_localMatrix);
    *localMatrixFabric = UsdUtil::glmToUsdMatrix(DEFAULT_MATRIX);

    for (uint64_t i = 0; i < texcoordSetCount; ++i) {
        const auto texcoordInterpolationFabric =
            fabricStage.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvars_st_interpolation_n(i));
        *texcoordInterpolationFabric = FabricTokens::vertex;
    }

    for (uint64_t i = 0; i < customVertexAttributesCount; ++i) {
        const auto& customVertexAttribute = CppUtil::getElementByIndex(customVertexAttributes, i);
        const auto customVertexAttributeInterpolationFabric =
            fabricStage.getAttributeWr<omni::fabric::TokenC>(_path, customVertexAttribute.fabricInterpolationName);
        *customVertexAttributeInterpolationFabric = FabricTokens::vertex;
    }

    if (hasNormals) {
        const auto normalsInterpolationFabric =
            fabricStage.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::normals_interpolation);
        *normalsInterpolationFabric = FabricTokens::vertex;
    }

    if (hasVertexColors) {
        const auto vertexColorInterpolationFabric =
            fabricStage.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvars_COLOR_0_interpolation);
        *vertexColorInterpolationFabric = FabricTokens::vertex;
    }

    if (hasVertexIds) {
        const auto vertexIdInterpolationFabric =
            fabricStage.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvars_vertexId_interpolation);
        *vertexIdInterpolationFabric = FabricTokens::vertex;
    }
}

void FabricGeometry::reset() {
    const auto hasNormals = _geometryDescriptor.hasNormals();
    const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
    const auto texcoordSetCount = _geometryDescriptor.getTexcoordSetCount();
    const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
    const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

    auto& fabricStage = _pContext->getFabricStage();

    // clang-format off
    const auto doubleSidedFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::doubleSided);
    const auto extentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    const auto worldExtentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    const auto worldVisibilityFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility);
    const auto gltfLocalToEcefTransformFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_gltfLocalToEcefTransform);
    const auto worldMatrixFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::omni_fabric_worldMatrix);

    const auto tilesetIdFabric = fabricStage.getAttributeWr<int64_t>(_path, FabricTokens::_cesium_tilesetId);
    // clang-format on

    *doubleSidedFabric = DEFAULT_DOUBLE_SIDED;
    *extentFabric = UsdUtil::glmToUsdExtent(DEFAULT_EXTENT);
    *worldExtentFabric = UsdUtil::glmToUsdExtent(DEFAULT_EXTENT);
    *worldVisibilityFabric = DEFAULT_VISIBILITY;
    *gltfLocalToEcefTransformFabric = UsdUtil::glmToUsdMatrix(DEFAULT_MATRIX);
    *worldMatrixFabric = UsdUtil::glmToUsdMatrix(DEFAULT_MATRIX);
    *tilesetIdFabric = FabricUtil::NO_TILESET_ID;

    fabricStage.setArrayAttributeSize(_path, FabricTokens::material_binding, 0);
    fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, 0);
    fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, 0);
    fabricStage.setArrayAttributeSize(_path, FabricTokens::points, 0);

    for (uint64_t i = 0; i < texcoordSetCount; ++i) {
        fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_st_n(i), 0);
    }

    for (const auto& customVertexAttribute : customVertexAttributes) {
        fabricStage.setArrayAttributeSize(_path, customVertexAttribute.fabricAttributeName, 0);
    }

    if (hasNormals) {
        fabricStage.setArrayAttributeSize(_path, FabricTokens::normals, 0);
    }

    if (hasVertexColors) {
        fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0, 0);
    }

    if (hasVertexIds) {
        fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId, 0);
    }
}

void FabricGeometry::setGeometry(
    int64_t tilesetId,
    const glm::dmat4& ecefToPrimWorldTransform,
    const glm::dmat4& gltfLocalToEcefTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    const FabricMaterialInfo& materialInfo,
    bool smoothNormals,
    double pointSize,
    const std::unordered_map<uint64_t, uint64_t>& texcoordIndexMapping,
    const std::unordered_map<uint64_t, uint64_t>& rasterOverlayTexcoordIndexMapping) {

    if (stageDestroyed()) {
        return;
    }

    const auto hasNormals = _geometryDescriptor.hasNormals();
    const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
    const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
    const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

    auto& fabricStage = _pContext->getFabricStage();

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto indices = GltfUtil::getIndices(model, primitive, positions);
    const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
    const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);
    const auto vertexIds = GltfUtil::getVertexIds(positions);
    const auto gltfLocalExtent = GltfUtil::getExtent(model, primitive);
    const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

    if (positions.size() == 0 || indices.size() == 0 || !gltfLocalExtent.has_value()) {
        return;
    }

    const auto doubleSided = materialInfo.doubleSided;
    const auto gltfLocalToPrimWorldTransform = ecefToPrimWorldTransform * gltfLocalToEcefTransform;
    const auto primWorldExtent = MathUtil::transformExtent(gltfLocalExtent.value(), gltfLocalToPrimWorldTransform);

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::POINTS) {
        const auto numVoxels = positions.size();
        const auto shapeHalfSize = (pointSize <= 0.0 ? 1.0 : pointSize) * 0.5;
        fabricStage.setArrayAttributeSize(_path, FabricTokens::points, numVoxels * 8);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, numVoxels * 2 * 6);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, numVoxels * 6 * 2 * 3);

        const auto pointsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points);
        const auto faceVertexCountsFabric = fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts);
        const auto faceVertexIndicesFabric =
            fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices);

        if (hasVertexColors) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0, numVoxels * 8);
            const auto vertexColorsFabric =
                fabricStage.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::primvars_COLOR_0);
            vertexColors.fill(vertexColorsFabric, 8);
        }

        if (hasVertexIds) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId, numVoxels * 8);
            const auto vertexIdsFabric = fabricStage.getArrayAttributeWr<float>(_path, FabricTokens::primvars_vertexId);
            vertexIds.fill(vertexIdsFabric, 8);
        }

        for (const auto& customVertexAttribute : customVertexAttributes) {
            CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE(
                setVertexAttributeValues,
                customVertexAttribute.type,
                fabricStage,
                _path,
                model,
                primitive,
                customVertexAttribute,
                uint64_t(8));
        }

        uint64_t vertIndex = 0;
        uint64_t vertexCountsIndex = 0;
        uint64_t faceVertexIndex = 0;
        for (uint64_t voxelIndex = 0; voxelIndex < numVoxels; ++voxelIndex) {
            const auto& center = positions.get(voxelIndex);

            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, -shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, -shapeHalfSize, -shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, -shapeHalfSize, shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{-shapeHalfSize, shapeHalfSize, shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, shapeHalfSize, shapeHalfSize} + center;
            pointsFabric[vertIndex++] = glm::fvec3{shapeHalfSize, -shapeHalfSize, shapeHalfSize} + center;

            for (int i = 0; i < 6; ++i) {
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
        fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, faceVertexCounts.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, indices.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::points, positions.size());

        const auto faceVertexCountsFabric = fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts);
        const auto faceVertexIndicesFabric =
            fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices);
        const auto pointsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points);

        faceVertexCounts.fill(faceVertexCountsFabric);
        indices.fill(faceVertexIndicesFabric);
        positions.fill(pointsFabric);

        const auto fillTexcoords = [this, &fabricStage](uint64_t texcoordIndex, const TexcoordsAccessor& texcoords) {
            assert(texcoordIndex < _geometryDescriptor.getTexcoordSetCount());
            const auto& primvarStToken = FabricTokens::primvars_st_n(texcoordIndex);
            fabricStage.setArrayAttributeSize(_path, primvarStToken, texcoords.size());
            const auto stFabric = fabricStage.getArrayAttributeWr<glm::fvec2>(_path, primvarStToken);
            texcoords.fill(stFabric);
        };

        for (const auto& [gltfSetIndex, primvarStIndex] : texcoordIndexMapping) {
            const auto texcoords = GltfUtil::getTexcoords(model, primitive, gltfSetIndex);
            fillTexcoords(primvarStIndex, texcoords);
        }

        for (const auto& [gltfSetIndex, primvarStIndex] : rasterOverlayTexcoordIndexMapping) {
            const auto texcoords = GltfUtil::getRasterOverlayTexcoords(model, primitive, gltfSetIndex);
            fillTexcoords(primvarStIndex, texcoords);
        }

        if (hasNormals) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::normals, normals.size());

            const auto normalsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::normals);

            normals.fill(normalsFabric);
        }

        if (hasVertexColors) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0, vertexColors.size());

            const auto vertexColorsFabric =
                fabricStage.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::primvars_COLOR_0);

            vertexColors.fill(vertexColorsFabric);
        }

        if (hasVertexIds) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId, vertexIds.size());

            const auto vertexIdsFabric = fabricStage.getArrayAttributeWr<float>(_path, FabricTokens::primvars_vertexId);

            vertexIds.fill(vertexIdsFabric);
        }

        for (const auto& customVertexAttribute : customVertexAttributes) {
            CALL_TEMPLATED_FUNCTION_WITH_RUNTIME_DATA_TYPE(
                setVertexAttributeValues,
                customVertexAttribute.type,
                fabricStage,
                _path,
                model,
                primitive,
                customVertexAttribute,
                uint64_t(1));
        }
    }

    // clang-format off
    const auto doubleSidedFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::doubleSided);
    const auto extentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    const auto worldExtentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    const auto gltfLocalToEcefTransformFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_gltfLocalToEcefTransform);
    const auto worldMatrixFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::omni_fabric_worldMatrix);
    const auto tilesetIdFabric = fabricStage.getAttributeWr<int64_t>(_path, FabricTokens::_cesium_tilesetId);
    // clang-format on

    *doubleSidedFabric = doubleSided;
    *extentFabric = UsdUtil::glmToUsdExtent(gltfLocalExtent.value());
    *worldExtentFabric = UsdUtil::glmToUsdExtent(primWorldExtent);
    *gltfLocalToEcefTransformFabric = UsdUtil::glmToUsdMatrix(gltfLocalToEcefTransform);
    *worldMatrixFabric = UsdUtil::glmToUsdMatrix(gltfLocalToPrimWorldTransform);

    *tilesetIdFabric = tilesetId;
}

bool FabricGeometry::stageDestroyed() {
    // Tile render resources may be processed asynchronously even after the tileset and stage have been destroyed.
    // Add this check to all public member functions, including constructors and destructors, to prevent them from
    // modifying the stage.
    return _stageId != _pContext->getUsdStageId();
}

}; // namespace cesium::omniverse
