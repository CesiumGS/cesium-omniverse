#include "cesium/omniverse/FabricGeometry.h"

#include "cesium/omniverse/CudaManager.h"
#include "cesium/omniverse/FabricAttributesBuilder.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricResourceManager.h"
#include "cesium/omniverse/FabricUtil.h"
#include "cesium/omniverse/GltfUtil.h"
#include "cesium/omniverse/Tokens.h"
#include "cesium/omniverse/UsdUtil.h"

#include <glm/fwd.hpp>
#include <string>

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
    attributes.addAttribute(FabricTypes::_cesium_tileId, FabricTokens::_cesium_tileId);
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
    auto subdivisionSchemeFabric = srw.getAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::subdivisionScheme);
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
    auto primvarsFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvars);
    auto primvarInterpolationsFabric = srw.getArrayAttributeWr<omni::fabric::TokenC>(_path, FabricTokens::primvarInterpolations);
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

    FabricUtil::setTilesetIdAndTileId(_path, NO_TILESET_ID, NO_TILE_ID);

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
    int64_t tileId,
    const glm::dmat4& ecefToUsdTransform,
    const glm::dmat4& gltfToEcefTransform,
    const glm::dmat4& nodeTransform,
    const CesiumGltf::Model& model,
    const CesiumGltf::MeshPrimitive& primitive,
    bool smoothNormals,
    bool hasImagery,
    float geometricError) {

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

    if (primitive.mode == CesiumGltf::MeshPrimitive::Mode::POINTS) {
        const int numberOfPointsPerShape = 4;
        const int numberOfFacesPerShape = 2;

        auto stageReaderWriter = Context::instance().getFabricStageReaderWriter();
        auto tileToken = CudaManager::getInstance().getTileToken(tileId);
        // DEBUG
        stageReaderWriter.createAttribute(_path, tileToken, CudaManager::getInstance().getTileTokenType());
        auto tileTokenAttribute = stageReaderWriter.getAttributeWr<double>(_path, tileToken);
        *tileTokenAttribute = 123.45;

        // how many quads, spheres, voxels, etc.
        const auto numberOfShapes = positions.size();
        geometricError = 100.0f; // TODO: dev
        const auto shapeHalfSize = 0.02f * geometricError;
        srw.setArrayAttributeSize(_path, FabricTokens::points, static_cast<size_t>(numberOfShapes * numberOfPointsPerShape));
        srw.setArrayAttributeSize(_path, FabricTokens::faceVertexCounts, numberOfShapes * numberOfFacesPerShape);
        srw.setArrayAttributeSize(_path, FabricTokens::faceVertexIndices, numberOfShapes * numberOfFacesPerShape * 3);

        auto pointsFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_path, FabricTokens::points);
        auto faceVertexCountsFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexCounts);
        auto faceVertexIndicesFabric = srw.getArrayAttributeWr<int>(_path, FabricTokens::faceVertexIndices);

        std::vector<glm::fvec3> vertexColorsData(numberOfShapes * numberOfPointsPerShape); // TODO: not great to waste this memory
        gsl::span<glm::fvec3> vertexColorsSpan(vertexColorsData);
        if (hasVertexColors) {
            vertexColors.fill(vertexColorsSpan);
            srw.setArrayAttributeSize(_path, FabricTokens::primvars_vertexColor, numberOfShapes * numberOfPointsPerShape);
        }
        auto vertexColorsFabric = srw.getArrayAttributeWr<glm::fvec3>(_path, FabricTokens::primvars_vertexColor);
        CudaKernelArgs kernelArgs;
        // (quad** quads, double3* lookatPosition, float3* lookatUp, float *quadSize, int numQuads)
        kernelArgs.args["quads"] = pointsFabric;
        kernelArgs.args["lookatPosition"] = glm::dvec3(0, 0, 0);
        kernelArgs.args["quadSize"] = 1.0f;
        kernelArgs.args["numQuads"] = numberOfShapes;

        size_t vertexIndex = 0;
        size_t faceVertexCountsIndex = 0;
        size_t faceVertexIndex = 0;
        size_t vertexColorsIndex = 0;
        for (size_t shapeIndex = 0; shapeIndex < numberOfShapes; shapeIndex++) {
            const auto& center = positions.get(shapeIndex);
            auto centerPxr = pxr::GfVec3f{center.x, center.y, center.z};

            pointsFabric[vertexIndex++] = pxr::GfVec3f{-shapeHalfSize, -shapeHalfSize, 0} + centerPxr;
            pointsFabric[vertexIndex++] = pxr::GfVec3f{-shapeHalfSize, shapeHalfSize, 0} + centerPxr;
            pointsFabric[vertexIndex++] = pxr::GfVec3f{shapeHalfSize, shapeHalfSize, 0} + centerPxr;
            pointsFabric[vertexIndex++] = pxr::GfVec3f{shapeHalfSize, -shapeHalfSize, 0} + centerPxr;

            for (int i = 0; i < numberOfFacesPerShape; i++) {
                faceVertexCountsFabric[faceVertexCountsIndex++] = 3;
            }

            const int vertexIndexOffset = static_cast<int>(shapeIndex * numberOfPointsPerShape);
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + vertexIndexOffset;
            faceVertexIndicesFabric[faceVertexIndex++] = 1 + vertexIndexOffset;
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + vertexIndexOffset;
            faceVertexIndicesFabric[faceVertexIndex++] = 0 + vertexIndexOffset;
            faceVertexIndicesFabric[faceVertexIndex++] = 2 + vertexIndexOffset;
            faceVertexIndicesFabric[faceVertexIndex++] = 3 + vertexIndexOffset;

            if (hasVertexColors) {
                const auto& color = vertexColorsSpan[shapeIndex];
                for (int i = 0; i < numberOfPointsPerShape; i++) {
                    vertexColorsFabric[vertexColorsIndex++] = color;
                }
            }
        }

        // std::cout << "loading tile " << tileId << std::endl;

        if (tileId == 0) {
            auto elementCount = pointsFabric.size();
            CudaManager::getInstance().createRunner(
            CudaKernelType::PRINT_POINTS,
            CudaUpdateType::ON_UPDATE_FRAME,
            tileId,
            kernelArgs,
            static_cast<int>(elementCount));

            // create a custom attribute for testing
            // stageReaderWriter.createAttribute(_path, getCudaTestAttributeFabricToken(), cudaTestAttributeFabricType);
            // auto testAttribute = stageReaderWriter.getAttributeWr<double>(_path, getCudaTestAttributeFabricToken());
            // *testAttribute = 123.45;

            // runProofOfConceptCode();
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

    // clang-format off
    auto extentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::extent);
    auto worldExtentFabric = srw.getAttributeWr<pxr::GfRange3d>(_path, FabricTokens::_worldExtent);
    auto localToEcefTransformFabric = srw.getAttributeWr<pxr::GfMatrix4d>(_path, FabricTokens::_cesium_localToEcefTransform);
    auto worldPositionFabric = srw.getAttributeWr<pxr::GfVec3d>(_path, FabricTokens::_worldPosition);
    auto worldOrientationFabric = srw.getAttributeWr<pxr::GfQuatf>(_path, FabricTokens::_worldOrientation);
    auto worldScaleFabric = srw.getAttributeWr<pxr::GfVec3f>(_path, FabricTokens::_worldScale);
    auto displayColorFabric = srw.getArrayAttributeWr<pxr::GfVec3f>(_path, FabricTokens::primvars_displayColor);
    auto displayOpacityFabric = srw.getArrayAttributeWr<float>(_path, FabricTokens::primvars_displayOpacity);
    // clang-format on

    *extentFabric = localExtent;
    *worldExtentFabric = worldExtent;
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

    FabricUtil::setTilesetIdAndTileId(_path, tilesetId, tileId);
}

bool FabricGeometry::stageDestroyed() {
    // Add this guard to all public member functions, including constructors and destructors. Tile render resources can
    // continue to be processed asynchronously even after the tileset and USD stage have been destroyed, so prevent any
    // operations that would modify the stage.
    return _stageId != UsdUtil::getUsdStageId();
}

// void FabricGeometry::runProofOfConceptCode() {
//     glm::fvec3 target{0, 0, 0};
//     glm::fvec3 targetUp{0, 1.0f, 0};
//     billboardMultiQuadCuda(target, targetUp);
// }

// void FabricGeometry::billboardMultiQuadCuda(glm::fvec3 lookatPosition, glm::fvec3 lookatUp) {
//     glm::dvec3 lookatPositionHost{0.0, 0.0, 0.0};
//     glm::fvec3 lookatUpHost{0.0, 1.0, 0.0};

//     lookatPositionHost.x = static_cast<double>(lookatPosition.x);
//     lookatPositionHost.y = static_cast<double>(lookatPosition.y);
//     lookatPositionHost.z = static_cast<double>(lookatPosition.z);

//     lookatUpHost.x = lookatUp.x;
//     lookatUpHost.y = lookatUp.y;
//     lookatUpHost.z = lookatUp.z;

//     //get all prims with the custom attr
//     auto iStageReaderWriter = carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
//     auto usdStageId = omni::fabric::UsdStageId(Context::instance().getStageId());
//     auto stageReaderWriterId = iStageReaderWriter->get(usdStageId);
//     auto stageReaderWriter = omni::fabric::StageReaderWriter(stageReaderWriterId);
//     omni::fabric::AttrNameAndType primTag(cudaTestAttributeFabricType, getCudaTestAttributeFabricToken());
//     auto bucketList = stageReaderWriter.findPrims({primTag});

//     // cudaRunner.init(lookAtMultiquadKernelCode, "lookAtMultiquadKernel");

//     // CUresult err;
//     // CUdeviceptr lookatPositionDevice;

//     // err = cuMemAlloc(&lookatPositionDevice, sizeof(glm::dvec3));
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // err = cuMemcpyHtoD(lookatPositionDevice, &lookatPositionHost, sizeof(glm::dvec3));
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // CUdeviceptr lookatUpDevice;
//     // err = cuMemAlloc(&lookatUpDevice, sizeof(glm::fvec3));
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // err = cuMemcpyHtoD(lookatUpDevice, &lookatUpHost, sizeof(glm::fvec3));
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // CUdeviceptr quadSizeDevice;
//     // err = cuMemAlloc(&quadSizeDevice, sizeof(float));
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemAlloc failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // err = cuMemcpyHtoD(quadSizeDevice, &_quadSizeHost, sizeof(float));
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemcpyHtoD failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // std::unordered_map<size_t, quad*> bucketQuadsPtrsMap;

//     // std::cout << "numBuckets " << bucketList.bucketCount() << std::endl;
//     for (size_t bucketNum = 0; bucketNum != bucketList.bucketCount(); bucketNum++)
//     {
//         // auto numQuadsSpan = stageReaderWriter.getAttributeArray<int>(bucketList, bucketNum, getNumQuadsAttributeFabricToken());
//         // int numQuads = numQuadsSpan[0];

//         // if (bucketQuadsPtrsMap.find(bucketNum) == bucketQuadsPtrsMap.end()) {
//         auto positions = stageReaderWriter.getAttributeArrayGpu<pxr::GfVec3f*>(bucketList, bucketNum, FabricTokens::points);
//         if (positions.data() != nullptr) {
//             std::cout << "no issue running PoC code" << std::endl;
//         }
//         // auto quadsPtr = reinterpret_cast<quad*>(positions.data());
//             // bucketQuadsPtrsMap[bucketNum] = quadsPtr;

//         // }

//         // int elemCount = numQuads;
//         // if (elemCount == 0) {
//         //     throw std::runtime_error("Fabric did not retrieve any elements");
//         // }
//         // // std::cout << elemCount << std::endl;
//         // void *args[] = { &bucketQuadsPtrsMap[0], &lookatPositionDevice, &lookatUpDevice, &quadSizeDevice, &elemCount}; //NOLINT

//         // cudaRunner.runKernel(args, static_cast<size_t>(elemCount));

//         // primCount += static_cast<int>(elemCount);
//     }

//     // std::cout << "modified " << primCount << " quads" << std::endl;

//     // err = cuMemFree(lookatPositionDevice);
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // err = cuMemFree(lookatUpDevice);
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }

//     // err = cuMemFree(quadSizeDevice);
//     // if (err != CUDA_SUCCESS) {
//     //     const char *errName;
//     //     const char *errStr;
//     //     cuGetErrorName(err, &errName);
//     //     cuGetErrorString(err, &errStr);
//     //     printf("cuMemFree failed: %s: %s\n", errName, errStr);
//     //     return;
//     // }
// }

}; // namespace cesium::omniverse
