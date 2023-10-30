#include "tilesetTests.h"

#include "testUtils.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#include <carb/dictionary/DictionaryUtils.h>
#include <carb/events/IEvents.h>
#include <doctest/doctest.h>
#include <omni/kit/IApp.h>

pxr::SdfPath endToEndTilesetPath;
bool endToEndTilesetLoaded = false;
carb::events::ISubscriptionPtr endToEndTilesetSubscriptionPtr;
class TilesetLoadListener;
TilesetLoadListener* tilesetLoadListener;

using namespace cesium::omniverse;

class TilesetLoadListener final : public carb::events::IEventListener {
  public:
    size_t refCount = 0;
    void onEvent(carb::events::IEvent* e [[maybe_unused]]) override {
        endToEndTilesetLoaded = true;
    };
    size_t addRef() override {
        return ++refCount;
    };
    size_t release() override {
        return --refCount;
    };
};

void setUpTilesetTests(const pxr::SdfPath& rootPath) {
    // Create a listener for tileset load events
    auto app = carb::getCachedInterface<omni::kit::IApp>();
    auto bus = app->getMessageBusEventStream();
    auto tilesetLoadedEvent = carb::events::typeFromString("cesium.omniverse.TILESET_LOADED");
    tilesetLoadListener = new TilesetLoadListener();
    endToEndTilesetSubscriptionPtr = bus->createSubscriptionToPushByType(tilesetLoadedEvent, tilesetLoadListener);

    // Load a local test tileset
    endToEndTilesetPath = UsdUtil::getPathUnique(rootPath, "endToEndTileset");
    auto endToEndTileset = UsdUtil::defineCesiumTileset(endToEndTilesetPath);
    std::string tilesetFilePath = "file://" TEST_WORKING_DIRECTORY "/tests/testAssets/tilesets/Tileset/tileset.json";

    endToEndTileset.GetSourceTypeAttr().Set(pxr::TfToken("url"));
    endToEndTileset.GetUrlAttr().Set(tilesetFilePath);
}
void cleanUpTilesetTests(const pxr::UsdStageRefPtr& stage) {
    endToEndTilesetSubscriptionPtr->unsubscribe();
    stage->RemovePrim(endToEndTilesetPath);
    delete tilesetLoadListener;
}

TEST_SUITE("Tileset tests") {
    TEST_CASE("End to end test") {

        // set by the TilesetLoadListener when any tileset successfully loads
        CHECK(endToEndTilesetLoaded);
    }
}
