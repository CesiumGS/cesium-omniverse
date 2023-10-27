#include "tilesetTests.h"

#include "testUtils.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/events/IEvents.h>
#include <doctest/doctest.h>
#include <omni/kit/IApp.h>

pxr::SdfPath endToEndTilesetPath;

using namespace cesium::omniverse;

void setUpTilesetTests(const pxr::SdfPath& rootPath) {
    auto app = carb::getCachedInterface<omni::kit::IApp>();
    auto bus = app->getMessageBusEventStream();

    // bus->createSubscriptionToPop(IEventListener *listener)

    endToEndTilesetPath = UsdUtil::getPathUnique(rootPath, "endToEndTileset");

    auto endToEndTileset = UsdUtil::defineCesiumTileset(endToEndTilesetPath);

    std::string tilesetFilePath = "file://" TEST_WORKING_DIRECTORY "/tests/testAssets/tilesets/Tileset/tileset.json";

    endToEndTileset.GetSourceTypeAttr().Set(pxr::TfToken("url"));
    endToEndTileset.GetUrlAttr().Set(tilesetFilePath);
}
void cleanUpTilesetTests(const pxr::UsdStageRefPtr& stage) {

    stage->RemovePrim(endToEndTilesetPath);
}

TEST_SUITE("Tileset tests") {
    TEST_CASE("End to end test") {

        //const auto& paths = AssetRegistry::getInstance().getAllTilesetPaths();
        const auto& tilesets = AssetRegistry::getInstance().getAllTilesets();

        CHECK(tilesets.size() > 0);

        // auto endToEndTilesetOptional = AssetRegistry::getInstance().getTilesetByPath(endToEndTilesetPath);
        // REQUIRE(endToEndTilesetOptional.has_value());

        // auto endToEndTileset = endToEndTilesetOptional.value().get();

        // CHECK_FALSE(endToEndTileset->activeLoading);
    }
}
