#include "tilesetTests.h"

#include "testUtils.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/events/IEvents.h>
#include <doctest/doctest.h>
#include <omni/kit/IApp.h>

#include <cstddef>

pxr::SdfPath endToEndTilesetPath;
bool endToEndTilesetLoaded = false;

using namespace cesium::omniverse;

class TilesetLoadListener final : public carb::events::IEventListener {
  public:
    size_t refCount = 0;
    void onEvent(carb::events::IEvent* e) override {
        // TODO remove
        e->addRef();
        endToEndTilesetLoaded = true;
        e->release();
    };
    size_t addRef() override {
        return ++refCount;
    };
    size_t release() override {
        return --refCount;
    };
};

auto t = TilesetLoadListener();

void setUpTilesetTests(const pxr::SdfPath& rootPath) {
    auto app = carb::getCachedInterface<omni::kit::IApp>();
    auto bus = app->getMessageBusEventStream();
    auto tilesetLoadedEvent = carb::events::typeFromString("cesium.omniverse.TILESET_LOADED");
    bus->createSubscriptionToPopByType(tilesetLoadedEvent, &t);

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

        // set by the TilesetLoadListener
        CHECK(endToEndTilesetLoaded);

        // auto endToEndTilesetOptional = AssetRegistry::getInstance().getTilesetByPath(endToEndTilesetPath);
        // REQUIRE(endToEndTilesetOptional.has_value());

        // auto endToEndTileset = endToEndTilesetOptional.value().get();

        // CHECK_FALSE(endToEndTileset->activeLoading);
    }
}
