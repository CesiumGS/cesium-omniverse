#include "cesium/omniverse/FabricGeometry.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/CppUtil.h"
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

#include <carb/Numeric.h>
#include <glm/fwd.hpp>

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <omni/fabric/FabricUSD.h>
#include <omni/fabric/SimStageWithHistory.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/base/gf/range3d.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace cesium::omniverse {

namespace {

const auto DEFAULT_DOUBLE_SIDED = false;
const auto DEFAULT_EXTENT = std::array<glm::dvec3, 2>{{glm::dvec3(0.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 0.0)}};
const auto DEFAULT_POSITION = glm::dvec3(0.0, 0.0, 0.0);
const auto DEFAULT_ORIENTATION = glm::dquat(1.0, 0.0, 0.0, 0.0);
const auto DEFAULT_SCALE = glm::dvec3(1.0, 1.0, 1.0);
const auto DEFAULT_MATRIX = glm::dmat4(1.0);
const auto DEFAULT_VISIBILITY = false;
const auto DEFAULT_SPHERICAL_HARMONICS_DEGREE = 0;

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
    accessor.fill(CppUtil::toSpan(fabricValues), repeat);
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

    const auto worldVisibilityFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility());
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
    fabricStage.setArrayAttributeSize(_path, FabricTokens::material_binding(), 1);
    const auto materialBindingFabric =
        fabricStage.getArrayAttributeWr<omni::fabric::Path>(_path, FabricTokens::material_binding());
    materialBindingFabric[0] = materialPath;
}

void FabricGeometry::initialize() {
    auto& fabricStage = _pContext->getFabricStage();
    fabricStage.createPrim(_path);

    //--------------------------------------------------------------------------
    // Declare attributes
    //--------------------------------------------------------------------------
    FabricAttributesBuilder attributes(_pContext);

    // Add attributes common to Gaussian Splats and meshes. Both inherit from
    // Gprim so add the Gprim class hierarchy and both use primvars so add
    // PrimvarsAPI.

    // clang-format off
    attributes.addAttribute(FabricTypes::extent, FabricTokens::extent());
    attributes.addAttribute(FabricTypes::_worldExtent, FabricTokens::_worldExtent());
    attributes.addAttribute(FabricTypes::_worldVisibility, FabricTokens::_worldVisibility());
    attributes.addAttribute(FabricTypes::_cesium_tilesetId, FabricTokens::_cesium_tilesetId());
    attributes.addAttribute(FabricTypes::_cesium_gltfLocalToEcefTransform, FabricTokens::_cesium_gltfLocalToEcefTransform());
    attributes.addAttribute(FabricTypes::omni_fabric_localMatrix, FabricTokens::omni_fabric_localMatrix());
    attributes.addAttribute(FabricTypes::omni_fabric_worldMatrix, FabricTokens::omni_fabric_worldMatrix());
    attributes.addAttribute(FabricTypes::Gprim, FabricTokens::Gprim());
    attributes.addAttribute(FabricTypes::Boundable, FabricTokens::Boundable());
    attributes.addAttribute(FabricTypes::Xformable, FabricTokens::Xformable());
    attributes.addAttribute(FabricTypes::Imageable, FabricTokens::Imageable());
    attributes.addAttribute(FabricTypes::Typed, FabricTokens::Typed());
    attributes.addAttribute(FabricTypes::SchemaBase, FabricTokens::SchemaBase());
    attributes.addAttribute(FabricTypes::TfType_Root, FabricTokens::TfType_Root());
    attributes.addAttribute(FabricTypes::PrimvarsAPI, FabricTokens::PrimvarsAPI());
    // clang-format on

    if (_geometryDescriptor.isGaussianSplats()) {
        // clang-format off
        attributes.addAttribute(FabricTypes::projectionModeHint, FabricTokens::projectionModeHint());
        attributes.addAttribute(FabricTypes::sortingModeHint, FabricTokens::sortingModeHint());
        attributes.addAttribute(FabricTypes::positions, FabricTokens::positions());
        attributes.addAttribute(FabricTypes::orientations, FabricTokens::orientations());
        attributes.addAttribute(FabricTypes::scales, FabricTokens::scales());
        attributes.addAttribute(FabricTypes::opacities, FabricTokens::opacities());
        attributes.addAttribute(FabricTypes::radiance_sphericalHarmonicsDegree, FabricTokens::radiance_sphericalHarmonicsDegree());
        attributes.addAttribute(FabricTypes::radiance_sphericalHarmonicsCoefficients, FabricTokens::radiance_sphericalHarmonicsCoefficients());
        attributes.addAttribute(FabricTypes::primvars_displayColor, FabricTokens::primvars_displayColor());
        attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::primvars_displayColor_interpolation());
        attributes.addAttribute(FabricTypes::ParticleField3DGaussianSplat, FabricTokens::ParticleField3DGaussianSplat());
        attributes.addAttribute(FabricTypes::ParticleField, FabricTokens::ParticleField());
        attributes.addAttribute(FabricTypes::ParticleFieldKernelBaseAPI, FabricTokens::ParticleFieldKernelBaseAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldKernelGaussianEllipsoidAPI, FabricTokens::ParticleFieldKernelGaussianEllipsoidAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldPositionAttributeAPI, FabricTokens::ParticleFieldPositionAttributeAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldPositionBaseAPI, FabricTokens::ParticleFieldPositionBaseAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldOpacityAttributeAPI, FabricTokens::ParticleFieldOpacityAttributeAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldOrientationAttributeAPI, FabricTokens::ParticleFieldOrientationAttributeAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldRadianceBaseAPI, FabricTokens::ParticleFieldRadianceBaseAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldScaleAttributeAPI, FabricTokens::ParticleFieldScaleAttributeAPI());
        attributes.addAttribute(FabricTypes::ParticleFieldSphericalHarmonicsAttributeAPI, FabricTokens::ParticleFieldSphericalHarmonicsAttributeAPI());
        // clang-format on
    } else {
        const auto hasNormals = _geometryDescriptor.hasNormals();
        const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
        const auto texcoordSetCount = _geometryDescriptor.getTexcoordSetCount();
        const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
        const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

        // clang-format off
        attributes.addAttribute(FabricTypes::faceVertexCounts, FabricTokens::faceVertexCounts());
        attributes.addAttribute(FabricTypes::faceVertexIndices, FabricTokens::faceVertexIndices());
        attributes.addAttribute(FabricTypes::points, FabricTokens::points());
        attributes.addAttribute(FabricTypes::doubleSided, FabricTokens::doubleSided());
        attributes.addAttribute(FabricTypes::subdivisionScheme, FabricTokens::subdivisionScheme());
        attributes.addAttribute(FabricTypes::material_binding, FabricTokens::material_binding());
        attributes.addAttribute(FabricTypes::Mesh, FabricTokens::Mesh());
        attributes.addAttribute(FabricTypes::PointBased, FabricTokens::PointBased());
        attributes.addAttribute(FabricTypes::MaterialBindingAPI, FabricTokens::MaterialBindingAPI());
        // clang-format on

        for (uint64_t i = 0; i < texcoordSetCount; ++i) {
            attributes.addAttribute(FabricTypes::primvars_st, FabricTokens::primvars_st_n(i));
            attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::primvars_st_interpolation_n(i));
        }

        if (hasNormals) {
            attributes.addAttribute(FabricTypes::normals, FabricTokens::normals());
            attributes.addAttribute(FabricTypes::primvars_interpolation, FabricTokens::normals_interpolation());
        }

        if (hasVertexColors) {
            attributes.addAttribute(FabricTypes::primvars_COLOR_0, FabricTokens::primvars_COLOR_0());
            attributes.addAttribute(
                FabricTypes::primvars_interpolation, FabricTokens::primvars_COLOR_0_interpolation());
        }

        if (hasVertexIds) {
            attributes.addAttribute(FabricTypes::primvars_vertexId, FabricTokens::primvars_vertexId());
            attributes.addAttribute(
                FabricTypes::primvars_interpolation, FabricTokens::primvars_vertexId_interpolation());
        }

        for (const auto& customVertexAttribute : customVertexAttributes) {
            attributes.addAttribute(
                FabricUtil::getPrimvarType(customVertexAttribute.type), customVertexAttribute.fabricAttributeName);
            attributes.addAttribute(FabricTypes::primvars_interpolation, customVertexAttribute.fabricInterpolationName);
        }
    }

    attributes.createAttributes(_path);

    //--------------------------------------------------------------------------
    // Set values for immutable attributes
    //--------------------------------------------------------------------------
    if (_geometryDescriptor.isGaussianSplats()) {
        // clang-format off
        const auto projectionModeHintFabric = fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::projectionModeHint());
        const auto sortingModeHintFabric = fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::sortingModeHint());
        const auto primvarsDisplayColorInterpolationFabric = fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::primvars_displayColor_interpolation());
        // clang-format on

        *projectionModeHintFabric = FabricTokens::perspective();
        *sortingModeHintFabric = FabricTokens::cameraDistance();
        *primvarsDisplayColorInterpolationFabric = FabricTokens::vertex();
    } else {
        const auto hasNormals = _geometryDescriptor.hasNormals();
        const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
        const auto texcoordSetCount = _geometryDescriptor.getTexcoordSetCount();
        const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
        const auto customVertexAttributesCount = customVertexAttributes.size();
        const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

        const auto subdivisionSchemeFabric =
            fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::subdivisionScheme());
        *subdivisionSchemeFabric = FabricTokens::none();

        for (uint64_t i = 0; i < texcoordSetCount; ++i) {
            const auto texcoordInterpolationFabric =
                fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::primvars_st_interpolation_n(i));
            *texcoordInterpolationFabric = FabricTokens::vertex();
        }

        for (uint64_t i = 0; i < customVertexAttributesCount; ++i) {
            const auto& customVertexAttribute = CppUtil::getElementByIndex(customVertexAttributes, i);
            const auto customVertexAttributeInterpolationFabric =
                fabricStage.getAttributeWr<omni::fabric::Token>(_path, customVertexAttribute.fabricInterpolationName);
            *customVertexAttributeInterpolationFabric = FabricTokens::vertex();
        }

        if (hasNormals) {
            const auto normalsInterpolationFabric =
                fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::normals_interpolation());
            *normalsInterpolationFabric = FabricTokens::vertex();
        }

        if (hasVertexColors) {
            const auto vertexColorInterpolationFabric =
                fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::primvars_COLOR_0_interpolation());
            *vertexColorInterpolationFabric = FabricTokens::vertex();
        }

        if (hasVertexIds) {
            const auto vertexIdInterpolationFabric =
                fabricStage.getAttributeWr<omni::fabric::Token>(_path, FabricTokens::primvars_vertexId_interpolation());
            *vertexIdInterpolationFabric = FabricTokens::vertex();
        }
    }
}

void FabricGeometry::reset() {
    auto& fabricStage = _pContext->getFabricStage();

    // clang-format off
    const auto extentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent());
    const auto worldExtentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent());
    const auto worldVisibilityFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::_worldVisibility());
    const auto gltfLocalToEcefTransformFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_gltfLocalToEcefTransform());
    const auto worldMatrixFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::omni_fabric_worldMatrix());
    const auto tilesetIdFabric = fabricStage.getAttributeWr<int64_t>(_path, FabricTokens::_cesium_tilesetId());
    // clang-format on

    *extentFabric = UsdUtil::glmToUsdExtent(DEFAULT_EXTENT);
    *worldExtentFabric = UsdUtil::glmToUsdExtent(DEFAULT_EXTENT);
    *worldVisibilityFabric = DEFAULT_VISIBILITY;
    *gltfLocalToEcefTransformFabric = UsdUtil::glmToUsdMatrix(DEFAULT_MATRIX);
    *worldMatrixFabric = UsdUtil::glmToUsdMatrix(DEFAULT_MATRIX);
    *tilesetIdFabric = FabricUtil::NO_TILESET_ID;

    if (_geometryDescriptor.isGaussianSplats()) {
        const auto radianceSphericalHarmonicsDegreeFabric =
            fabricStage.getAttributeWr<int>(_path, FabricTokens::radiance_sphericalHarmonicsDegree());

        *radianceSphericalHarmonicsDegreeFabric = DEFAULT_SPHERICAL_HARMONICS_DEGREE;

        fabricStage.setArrayAttributeSize(_path, FabricTokens::positions(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::orientations(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::scales(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::opacities(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::radiance_sphericalHarmonicsCoefficients(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_displayColor(), 0);
    } else {
        const auto hasNormals = _geometryDescriptor.hasNormals();
        const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
        const auto texcoordSetCount = _geometryDescriptor.getTexcoordSetCount();
        const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
        const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

        const auto doubleSidedFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::doubleSided());
        *doubleSidedFabric = DEFAULT_DOUBLE_SIDED;

        fabricStage.setArrayAttributeSize(_path, FabricTokens::material_binding(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices(), 0);
        fabricStage.setArrayAttributeSize(_path, FabricTokens::points(), 0);

        for (uint64_t i = 0; i < texcoordSetCount; ++i) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_st_n(i), 0);
        }

        for (const auto& customVertexAttribute : customVertexAttributes) {
            fabricStage.setArrayAttributeSize(_path, customVertexAttribute.fabricAttributeName, 0);
        }

        if (hasNormals) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::normals(), 0);
        }

        if (hasVertexColors) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0(), 0);
        }

        if (hasVertexIds) {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId(), 0);
        }
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

    const auto positions = GltfUtil::getPositions(model, primitive);
    const auto gltfLocalExtent = GltfUtil::getExtent(model, primitive);
    const auto gltfLocalToPrimWorldTransform = ecefToPrimWorldTransform * gltfLocalToEcefTransform;
    const auto primWorldExtent = MathUtil::transformExtent(gltfLocalExtent.value(), gltfLocalToPrimWorldTransform);

    if (positions.size() == 0 || !gltfLocalExtent.has_value()) {
        return;
    }

    auto& fabricStage = _pContext->getFabricStage();

    // clang-format off
    const auto extentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent());
    const auto worldExtentFabric = fabricStage.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent());
    const auto gltfLocalToEcefTransformFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_gltfLocalToEcefTransform());
    const auto localMatrixFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::omni_fabric_localMatrix());
    const auto worldMatrixFabric = fabricStage.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::omni_fabric_worldMatrix());
    const auto tilesetIdFabric = fabricStage.getAttributeWr<int64_t>(_path, FabricTokens::_cesium_tilesetId());
    // clang-format on

    *extentFabric = UsdUtil::glmToUsdExtent(gltfLocalExtent.value());
    *worldExtentFabric = UsdUtil::glmToUsdExtent(primWorldExtent);
    *gltfLocalToEcefTransformFabric = UsdUtil::glmToUsdMatrix(gltfLocalToEcefTransform);
    *localMatrixFabric = UsdUtil::glmToUsdMatrix(gltfLocalToPrimWorldTransform);
    *worldMatrixFabric = UsdUtil::glmToUsdMatrix(gltfLocalToPrimWorldTransform);
    *tilesetIdFabric = tilesetId;

    if (_geometryDescriptor.isGaussianSplats()) {
        const auto rotations = GltfUtil::getGaussianSplatRotations(model, primitive);
        const auto scales = GltfUtil::getGaussianSplatScales(model, primitive);
        const auto opacities = GltfUtil::getGaussianSplatOpacities(model, primitive);
        const auto sphericalHarmonicCoefficients =
            GltfUtil::getGaussianSplatSphericalHarmonicCoefficients(model, primitive);
        const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);

        const auto sphericalHarmonicDegrees = GltfUtil::getGaussianSplatSphericalHarmonicDegrees(primitive).value_or(0);

        const auto radianceSphericalHarmonicsDegreeFabric =
            fabricStage.getAttributeWr<int>(_path, FabricTokens::radiance_sphericalHarmonicsDegree());
        *radianceSphericalHarmonicsDegreeFabric = static_cast<int>(sphericalHarmonicDegrees);

        // clang-format off
        fabricStage.setArrayAttributeSize(_path, FabricTokens::positions(), positions.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::orientations(), rotations.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::scales(), scales.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::opacities(), opacities.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::radiance_sphericalHarmonicsCoefficients(), sphericalHarmonicCoefficients.size());
        fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_displayColor(), vertexColors.size());
        // clang-format on

        // clang-format off
        const auto positionsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::positions());
        const auto orientationsFabric = fabricStage.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::orientations());
        const auto scalesFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::scales());
        const auto opacitiesFabric = fabricStage.getArrayAttributeWr<float>(_path, FabricTokens::opacities());
        const auto radianceSphericalHarmonicCoefficientsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::radiance_sphericalHarmonicsCoefficients());
        const auto primvarsDisplayColorsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::primvars_displayColor());
        // clang-format on

        positions.fill(CppUtil::toSpan(positionsFabric));
        rotations.fill(CppUtil::toSpan(orientationsFabric));
        scales.fill(CppUtil::toSpan(scalesFabric));
        opacities.fill(CppUtil::toSpan(opacitiesFabric));
        sphericalHarmonicCoefficients.fill(CppUtil::toSpan(radianceSphericalHarmonicCoefficientsFabric));
        vertexColors.fill(CppUtil::toSpan(primvarsDisplayColorsFabric));
    } else {
        const auto hasNormals = _geometryDescriptor.hasNormals();
        const auto hasVertexColors = _geometryDescriptor.hasVertexColors();
        const auto& customVertexAttributes = _geometryDescriptor.getCustomVertexAttributes();
        const auto hasVertexIds = _geometryDescriptor.hasVertexIds();

        const auto indices = GltfUtil::getIndices(model, primitive, positions);
        const auto normals = GltfUtil::getNormals(model, primitive, positions, indices, smoothNormals);
        const auto vertexColors = GltfUtil::getVertexColors(model, primitive, 0);
        const auto vertexIds = GltfUtil::getVertexIds(positions);
        const auto faceVertexCounts = GltfUtil::getFaceVertexCounts(indices);

        if (indices.size() == 0) {
            return;
        }

        const auto doubleSidedFabric = fabricStage.getAttributeWr<bool>(_path, FabricTokens::doubleSided());
        *doubleSidedFabric = materialInfo.doubleSided;

        if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::POINTS) {
            // Fabric can't render OpenGL-style points so convert each point to a box
            const auto shapeHalfSize = (pointSize <= 0.0 ? 1.0 : pointSize) * 0.5;
            fabricStage.setArrayAttributeSize(_path, FabricTokens::points(), positions.size() * 8);
            fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts(), positions.size() * 2 * 6);
            fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices(), positions.size() * 6 * 2 * 3);

            // clang-format off
            const auto pointsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points());
            const auto faceVertexCountsFabric = fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts());
            const auto faceVertexIndicesFabric = fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices());
            // clang-format on

            if (hasVertexColors) {
                fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0(), positions.size() * 8);
                const auto vertexColorsFabric =
                    fabricStage.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::primvars_COLOR_0());
                vertexColors.fill(CppUtil::toSpan(vertexColorsFabric), 8);
            }

            if (hasVertexIds) {
                fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId(), positions.size() * 8);
                const auto vertexIdsFabric =
                    fabricStage.getArrayAttributeWr<float>(_path, FabricTokens::primvars_vertexId());
                vertexIds.fill(CppUtil::toSpan(vertexIdsFabric), 8);
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

            uint64_t vertexIndex = 0;
            uint64_t vertexCountsIndex = 0;
            uint64_t faceVertexIndex = 0;

            for (uint64_t pointIndex = 0; pointIndex < positions.size(); ++pointIndex) {
                const auto& center = positions.get(pointIndex);

                pointsFabric[vertexIndex++] = glm::fvec3{-shapeHalfSize, -shapeHalfSize, -shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{-shapeHalfSize, shapeHalfSize, -shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{shapeHalfSize, shapeHalfSize, -shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{shapeHalfSize, -shapeHalfSize, -shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{-shapeHalfSize, -shapeHalfSize, shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{-shapeHalfSize, shapeHalfSize, shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{shapeHalfSize, shapeHalfSize, shapeHalfSize} + center;
                pointsFabric[vertexIndex++] = glm::fvec3{shapeHalfSize, -shapeHalfSize, shapeHalfSize} + center;

                for (int i = 0; i < 6; ++i) {
                    faceVertexCountsFabric[vertexCountsIndex++] = 3;
                    faceVertexCountsFabric[vertexCountsIndex++] = 3;
                }

                // front
                faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(pointIndex * 8);
                // left
                faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(pointIndex * 8);
                // right
                faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(pointIndex * 8);
                // top
                faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 1 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 2 + static_cast<int>(pointIndex * 8);
                // bottom
                faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 3 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 0 + static_cast<int>(pointIndex * 8);
                // back
                faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 6 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 7 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 5 + static_cast<int>(pointIndex * 8);
                faceVertexIndicesFabric[faceVertexIndex++] = 4 + static_cast<int>(pointIndex * 8);
            }
        } else {
            fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts(), faceVertexCounts.size());
            fabricStage.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices(), indices.size());
            fabricStage.setArrayAttributeSize(_path, FabricTokens::points(), positions.size());

            // clang-format off
            const auto faceVertexCountsFabric = fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts());
            const auto faceVertexIndicesFabric = fabricStage.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices());
            const auto pointsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::points());
            // clang-format on

            faceVertexCounts.fill(CppUtil::toSpan(faceVertexCountsFabric));
            indices.fill(CppUtil::toSpan(faceVertexIndicesFabric));
            positions.fill(CppUtil::toSpan(pointsFabric));

            const auto fillTexcoords = [this,
                                        &fabricStage](uint64_t texcoordIndex, const TexcoordsAccessor& texcoords) {
                assert(texcoordIndex < _geometryDescriptor.getTexcoordSetCount());
                const auto& primvarStToken = FabricTokens::primvars_st_n(texcoordIndex);
                fabricStage.setArrayAttributeSize(_path, primvarStToken, texcoords.size());
                const auto stFabric = fabricStage.getArrayAttributeWr<glm::fvec2>(_path, primvarStToken);
                texcoords.fill(CppUtil::toSpan(stFabric));
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
                fabricStage.setArrayAttributeSize(_path, FabricTokens::normals(), normals.size());

                const auto normalsFabric = fabricStage.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::normals());

                normals.fill(CppUtil::toSpan(normalsFabric));
            }

            if (hasVertexColors) {
                fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_COLOR_0(), vertexColors.size());

                const auto vertexColorsFabric =
                    fabricStage.getArrayAttributeWr<glm::fvec4>(_path, FabricTokens::primvars_COLOR_0());

                vertexColors.fill(CppUtil::toSpan(vertexColorsFabric));
            }

            if (hasVertexIds) {
                fabricStage.setArrayAttributeSize(_path, FabricTokens::primvars_vertexId(), vertexIds.size());

                const auto vertexIdsFabric =
                    fabricStage.getArrayAttributeWr<float>(_path, FabricTokens::primvars_vertexId());

                vertexIds.fill(CppUtil::toSpan(vertexIdsFabric));
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
    }
}

bool FabricGeometry::stageDestroyed() {
    // Tile render resources may be processed asynchronously even after the tileset and stage have been destroyed.
    // Add this check to all public member functions, including constructors and destructors, to prevent them from
    // modifying the stage.
    return _stageId != _pContext->getUsdStageId();
}

}; // namespace cesium::omniverse
