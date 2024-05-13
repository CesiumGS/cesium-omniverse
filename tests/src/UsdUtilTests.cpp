#include "UsdUtilTests.h"

#include "testUtils.h"

#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/data.h>
#include <CesiumUsdSchemas/georeference.h>
#include <CesiumUsdSchemas/globeAnchorAPI.h>
#include <CesiumUsdSchemas/ionRasterOverlay.h>
#include <CesiumUsdSchemas/ionServer.h>
#include <CesiumUsdSchemas/tileset.h>
#include <doctest/doctest.h>
#include <glm/ext/matrix_double4x4.hpp>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/xformable.h>

// define prim paths globally to cut down on repeated definitions
// name the paths after the function to be tested so they can easily be paired up later
pxr::SdfPath defineCesiumDataPath;
pxr::SdfPath defineCesiumGeoreferencePath;
pxr::SdfPath defineCesiumTilesetPath;
pxr::SdfPath defineCesiumIonRasterOverlayPath;
pxr::SdfPath defineGlobeAnchorPath;

using namespace cesium::omniverse;
using namespace cesium::omniverse::UsdUtil;

const Context* pContext;

void setUpUsdUtilTests(cesium::omniverse::Context* context, const pxr::SdfPath& rootPath) {
    // might as well name the prims after the function as well, to ensure uniqueness and clarity
    defineCesiumDataPath = rootPath.AppendChild(pxr::TfToken("defineCesiumData"));
    defineCesiumGeoreferencePath = rootPath.AppendChild(pxr::TfToken("defineCesiumGeoreference"));
    defineCesiumIonRasterOverlayPath = rootPath.AppendChild(pxr::TfToken("defineCesiumIonRasterOverlay"));
    defineCesiumTilesetPath = rootPath.AppendChild(pxr::TfToken("defineCesiumTileset"));
    defineGlobeAnchorPath = rootPath.AppendChild(pxr::TfToken("defineGlobeAnchor"));

    defineCesiumData(context->getUsdStage(), defineCesiumDataPath);
    defineCesiumGeoreference(context->getUsdStage(), defineCesiumGeoreferencePath);
    defineCesiumTileset(context->getUsdStage(), defineCesiumTilesetPath);
    defineCesiumIonRasterOverlay(context->getUsdStage(), defineCesiumIonRasterOverlayPath);

    pContext = context;
}

void cleanUpUsdUtilTests(const pxr::UsdStageRefPtr& stage) {

    // might as well name the prims after the function as well, to ensure uniqueness and clarity
    stage->RemovePrim(defineCesiumDataPath);
    stage->RemovePrim(defineCesiumGeoreferencePath);
    stage->RemovePrim(defineCesiumIonRasterOverlayPath);
    stage->RemovePrim(defineCesiumTilesetPath);
    stage->RemovePrim(defineGlobeAnchorPath);
}

TEST_SUITE("UsdUtil tests") {

    TEST_CASE("Check expected initial state") {
        auto cesiumObjPath = pxr::SdfPath("/Cesium");
        CHECK(primExists(pContext->getUsdStage(), cesiumObjPath));
        // TODO can we check something invisible here too?
        CHECK(isPrimVisible(pContext->getUsdStage(), cesiumObjPath));
    }

    TEST_CASE("Check glm/usd conversion functions") {
        glm::dmat4 matrix(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

        // Round-trip conversion of usd/glm matrix
        CHECK(matrix == usdToGlmMatrix(glmToUsdMatrix(matrix)));
    }

    TEST_CASE("Tests that require prim creation") {
        auto primPath = makeUniquePath(pContext->getUsdStage(), getRootPath(pContext->getUsdStage()), "CesiumTestPrim");
        auto prim = pContext->getUsdStage()->DefinePrim(primPath);

        // Intentionally try the same prim name
        auto cubePath = makeUniquePath(pContext->getUsdStage(), getRootPath(pContext->getUsdStage()), "CesiumTestPrim");
        // Tests makeUniquePath actually returns unique paths
        CHECK(primPath.GetPrimPath() != cubePath.GetPrimPath());

        auto cube = pxr::UsdGeomCube::Define(pContext->getUsdStage(), cubePath);

        auto xformApiCube = pxr::UsdGeomXformCommonAPI(cube);
        xformApiCube.SetRotate({30, 60, 90});
        xformApiCube.SetScale({5, 12, 13});
        xformApiCube.SetTranslate({3, 4, 5});

        auto xformableCube = pxr::UsdGeomXformable(cube);

        pxr::GfMatrix4d cubeXform;
        bool xformStackResetNeeded [[maybe_unused]];

        xformableCube.GetLocalTransformation(&cubeXform, &xformStackResetNeeded);

        CHECK(usdToGlmMatrix(cubeXform) == computePrimLocalToWorldTransform(pContext->getUsdStage(), cubePath));

        pContext->getUsdStage()->RemovePrim(primPath);
        pContext->getUsdStage()->RemovePrim(cubePath);
    }

    TEST_CASE("Test UTF-8 path names") {
        for (int i = 0; i < NUM_TEST_REPETITIONS; ++i) {
            std::string randomUTF8String = "safe_name_test";

            randomUTF8String.reserve(64);

            for (long unsigned int ii = 0; ii < randomUTF8String.capacity() - randomUTF8String.size(); ++ii) {
                char randChar = (char)(rand() % 0xE007F);
                randomUTF8String.append(&randChar);
            }

            auto safeUniquePath =
                makeUniquePath(pContext->getUsdStage(), getRootPath(pContext->getUsdStage()), randomUTF8String);

            pContext->getUsdStage()->DefinePrim(safeUniquePath);
            CHECK(primExists(pContext->getUsdStage(), safeUniquePath));
            pContext->getUsdStage()->RemovePrim(safeUniquePath);
            CHECK_FALSE(primExists(pContext->getUsdStage(), safeUniquePath));
        }
    }

    TEST_CASE("Cesium helper functions") {

        auto rootPath = getRootPath(pContext->getUsdStage());

        CHECK(isCesiumData(pContext->getUsdStage(), defineCesiumDataPath));
        CHECK(isCesiumGeoreference(pContext->getUsdStage(), defineCesiumGeoreferencePath));
        CHECK(isCesiumTileset(pContext->getUsdStage(), defineCesiumTilesetPath));
        CHECK(isCesiumIonRasterOverlay(pContext->getUsdStage(), defineCesiumIonRasterOverlayPath));
    }

    TEST_CASE("Smoke tests") {
        // functions for which we do not yet have better tests,
        // but we can at least verify they don't throw
        CHECK_NOTHROW(getDynamicTextureProviderAssetPathToken("foo"));
        CHECK_NOTHROW(getUsdUpAxis(pContext->getUsdStage()));
        CHECK(getUsdMetersPerUnit(pContext->getUsdStage()) > 0);
    }
}
