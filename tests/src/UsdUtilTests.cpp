#include "UsdUtilTests.h"

#include "CesiumUsdSchemas/data.h"
#include "CesiumUsdSchemas/georeference.h"
#include "testUtils.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/LoggerSink.h"
#include "cesium/omniverse/UsdUtil.h"

#include <doctest/doctest.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/xformable.h>

// define prim paths globally to cut down on repeated definitions
// name the paths after the function to be tested so they can easily be paired up later
pxr::SdfPath defineCesiumDataPath;
pxr::SdfPath defineCesiumSessionPath;
pxr::SdfPath defineCesiumGeoreferencePath;
pxr::SdfPath defineCesiumTilesetPath;
pxr::SdfPath defineCesiumImageryPath;
pxr::SdfPath defineGlobeAnchorPath;
pxr::CesiumData getOrCreateCesiumDataPrim;
pxr::CesiumSession getOrCreateCesiumSessionPrim;
pxr::CesiumGeoreference getOrCreateCesiumGeoreferencePrim;

namespace cesium::omniverse::UsdUtil {
void setUpUsdUtilTests(long int stage_id) {

    Context::instance().setStageId(stage_id);

    auto stage = getUsdStage();
    auto rootPath = getRootPath();

    // might as well name the prims after the function as well, to ensure uniqueness and clarity
    defineCesiumDataPath = rootPath.AppendChild(pxr::TfToken("defineCesiumData"));
    defineCesiumSessionPath = rootPath.AppendChild(pxr::TfToken("defineCesiumSession"));
    defineCesiumGeoreferencePath = rootPath.AppendChild(pxr::TfToken("defineCesiumGeoreference"));
    defineCesiumImageryPath = rootPath.AppendChild(pxr::TfToken("defineCesiumImagery"));
    defineCesiumTilesetPath = rootPath.AppendChild(pxr::TfToken("defineCesiumTileset"));
    defineGlobeAnchorPath = rootPath.AppendChild(pxr::TfToken("defineGlobeAnchor"));

    defineCesiumData(defineCesiumDataPath);
    defineCesiumSession(defineCesiumSessionPath);
    defineCesiumGeoreference(defineCesiumGeoreferencePath);
    defineCesiumTileset(defineCesiumTilesetPath);
    defineCesiumImagery(defineCesiumImageryPath);
    // defineGlobeAnchor(globeAnchorPath);

    getOrCreateCesiumDataPrim = getOrCreateCesiumData();
    getOrCreateCesiumSessionPrim = getOrCreateCesiumSession();
    getOrCreateCesiumGeoreferencePrim = getOrCreateCesiumGeoreference();
}

void cleanUpUsdUtilTests() {

    auto stage = getUsdStage();
    auto rootPath = getRootPath();

    // might as well name the prims after the function as well, to ensure uniqueness and clarity
    stage->RemovePrim(defineCesiumDataPath);
    stage->RemovePrim(defineCesiumSessionPath);
    stage->RemovePrim(defineCesiumGeoreferencePath);
    stage->RemovePrim(defineCesiumImageryPath);
    stage->RemovePrim(defineCesiumTilesetPath);
    stage->RemovePrim(defineGlobeAnchorPath);

    // stage->RemovePrim(globeAnchorPath);

    stage->RemovePrim(getOrCreateCesiumDataPrim.GetPath());
    stage->RemovePrim(getOrCreateCesiumSessionPrim.GetPath());
    stage->RemovePrim(getOrCreateCesiumGeoreferencePrim.GetPath());
}

TEST_SUITE("UsdUtil tests") {

    TEST_CASE("Check expected initial state") {
        auto cesiumObjPath = pxr::SdfPath("/Cesium");
        CHECK(hasStage());
        CHECK(primExists(cesiumObjPath));
        // TODO can we check something invisible here too?
        CHECK(isPrimVisible(cesiumObjPath));
    }

    TEST_CASE("Check glm/usd conversion functions") {
        glm::dmat4 matrix(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        // Round-trip conversion of usd/glm matrix
        CHECK(matrix == usdToGlmMatrix(glmToUsdMatrix(matrix)));
    }

    TEST_CASE("Tests that require prim creation") {
        auto stage = getUsdStage();
        auto primPath = getPathUnique(getRootPath(), "CesiumTestPrim");
        auto prim = stage->DefinePrim(primPath);

        // Intentionally try the same prim name
        auto cubePath = getPathUnique(getRootPath(), "CesiumTestPrim");
        // Tests getPathUnique actually returns unique paths
        CHECK(primPath.GetPrimPath() != cubePath.GetPrimPath());

        auto cube = pxr::UsdGeomCube::Define(stage, cubePath);

        auto xformApiCube = pxr::UsdGeomXformCommonAPI(cube);
        xformApiCube.SetRotate({30, 60, 90});
        xformApiCube.SetScale({5, 12, 13});
        xformApiCube.SetTranslate({3, 4, 5});

        auto xformableCube = pxr::UsdGeomXformable(cube);

        pxr::GfMatrix4d cubeXform;
        bool xformStackResetNeeded [[maybe_unused]];

        xformableCube.GetLocalTransformation(&cubeXform, &xformStackResetNeeded);

        CHECK(usdToGlmMatrix(cubeXform) == computeUsdLocalToWorldTransform(cubePath));

        stage->RemovePrim(primPath);
        stage->RemovePrim(cubePath);
    }

    TEST_CASE("Test UTF-8 path names") {
        auto stage = getUsdStage();

        for (int i = 0; i < NUM_TEST_REPETITIONS; i++) {
            std::string randomUTF8String = "safe_name_test";

            randomUTF8String.reserve(64);

            for (long unsigned int ii = 0; ii < randomUTF8String.capacity() - randomUTF8String.size(); ii++) {
                char randChar = (char)(rand() % 0xE007F);
                randomUTF8String.append(&randChar);
            }

            auto safeUniquePath = getPathUnique(getRootPath(), getSafeName(randomUTF8String));

            stage->DefinePrim(safeUniquePath);
            CHECK(primExists(safeUniquePath));
            stage->RemovePrim(safeUniquePath);
            CHECK_FALSE(primExists(safeUniquePath));
        }
    }

    TEST_CASE("Cesium helper functions") {

        auto stage = getUsdStage();
        auto rootPath = getRootPath();

        CHECK(isCesiumData(defineCesiumDataPath));
        CHECK(isCesiumSession(defineCesiumSessionPath));
        CHECK(isCesiumGeoreference(defineCesiumGeoreferencePath));
        CHECK(isCesiumTileset(defineCesiumTilesetPath));
        CHECK(isCesiumImagery(defineCesiumImageryPath));
        // CHECK(hasCesiumGlobeAnchor(globeAnchorPath));

        CHECK(isCesiumData(getOrCreateCesiumDataPrim.GetPath()));
        CHECK(isCesiumSession(getOrCreateCesiumSessionPrim.GetPath()));
        CHECK(isCesiumGeoreference(getOrCreateCesiumGeoreferencePrim.GetPath()));
    }

    TEST_CASE("Smoke tests") {
        // functions for which we do not yet have better tests,
        // but we can at least verify they don't throw
        CHECK_NOTHROW(getDynamicTextureProviderAssetPathToken("foo"));
        CHECK_NOTHROW(getFabricStageReaderWriter());
        CHECK_NOTHROW(getFabricStageReaderWriterId());
        CHECK_NOTHROW(getUsdUpAxis());
        CHECK(getUsdMetersPerUnit() > 0);

        // List of currently untested functions in UsdUtil:
        // usdToGlmVector(const pxr::GfVec3d& vector);
        // glmToUsdVector(const glm::dvec3& vector);
        // glmToUsdVector(const glm::fvec2& vector);
        // glmToUsdVector(const glm::fvec3& vector);
        // glmToUsdRange(const std::array<glm::dvec3, 2>& range);
        // glmToUsdQuat(const glm::dquat& quat);
        // glmToUsdMatrixDecomposed(const glm::dmat4& matrix);
        // computeUsdWorldToLocalTransform(const pxr::SdfPath& path);
        // computeEcefToUsdLocalTransform(const CesiumGeospatial::Cartographic& origin);
        // computeEcefToUsdWorldTransformForPrim(
        //     const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath);
        // computeUsdWorldToEcefTransformForPrim(
        //     const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath);
        // computeEcefToUsdLocalTransformForPrim(
        //     const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath);
        // computeUsdLocalToEcefTransformForPrim(
        //     const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath);
        // computeViewState(
        //     const CesiumGeospatial::Cartographic& origin, const pxr::SdfPath& primPath, const Viewport& viewport);
        // computeWorldExtent(const pxr::GfRange3d& localExtent, const glm::dmat4& localToUsdTransform);
        // getEulerAnglesFromQuaternion(const pxr::GfQuatf& quaternion);
        // setGeoreferenceForTileset(const pxr::SdfPath& tilesetPath, const pxr::SdfPath& georeferencePath);
        // addOrUpdateTransformOpForAnchor(const pxr::SdfPath& path, const glm::dmat4& transform);
        // getCesiumTransformOpValueForPathIfExists(const pxr::SdfPath& path);
        // getAnchorGeoreferencePath(const pxr::SdfPath& path);
        // getCartographicOriginForAnchor(const pxr::SdfPath& path);
    }
}
} // namespace cesium::omniverse::UsdUtil
