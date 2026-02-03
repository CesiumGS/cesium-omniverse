#include "tilesetTests.h"

#include "testUtils.h"

#include "cesium/omniverse/AssetRegistry.h"
#include "cesium/omniverse/Context.h"
#include "cesium/omniverse/OmniTileset.h"
#include "cesium/omniverse/UsdUtil.h"

#include <CesiumUsdSchemas/tileset.h>
#include <carb/dictionary/DictionaryUtils.h>
#include <carb/events/IEvents.h>
#include <doctest/doctest.h>

// Suppress deprecation warning for getMessageBusEventStream until Events 2.0 migration is complete
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <omni/kit/IApp.h>

#include <memory>

pxr::SdfPath endToEndTilesetPath;
bool endToEndTilesetLoaded = false;
carb::events::ISubscriptionPtr endToEndTilesetSubscriptionPtr;
class TilesetLoadListener;
std::unique_ptr<TilesetLoadListener> tilesetLoadListener;

using namespace cesium::omniverse;

class TilesetLoadListener final : public carb::events::IEventListener {
  public:
    uint64_t refCount = 0;
    void onEvent(carb::events::IEvent* e [[maybe_unused]]) override {
        endToEndTilesetLoaded = true;
    };
    uint64_t addRef() override {
        return ++refCount;
    };
    uint64_t release() override {
        return --refCount;
    };
};

void setUpTilesetTests(Context* pContext, const pxr::SdfPath& rootPath) {
    // Create a listener for tileset load events
    auto app = carb::getCachedInterface<omni::kit::IApp>();
    auto bus = app->getMessageBusEventStream();
    auto tilesetLoadedEvent = carb::events::typeFromString("cesium.omniverse.TILESET_LOADED");
    tilesetLoadListener = std::make_unique<TilesetLoadListener>();
    endToEndTilesetSubscriptionPtr = bus->createSubscriptionToPushByType(tilesetLoadedEvent, tilesetLoadListener.get());

    // Load a local test tileset
    endToEndTilesetPath = UsdUtil::makeUniquePath(pContext->getUsdStage(), rootPath, "endToEndTileset");
    auto endToEndTileset = UsdUtil::defineCesiumTileset(pContext->getUsdStage(), endToEndTilesetPath);
    std::string tilesetFilePath = "file://" TEST_WORKING_DIRECTORY "/tests/testAssets/tilesets/Tileset/tileset.json";

    endToEndTileset.GetSourceTypeAttr().Set(pxr::TfToken("url"));
    endToEndTileset.GetUrlAttr().Set(tilesetFilePath);
}
void cleanUpTilesetTests(const pxr::UsdStageRefPtr& stage) {
    endToEndTilesetSubscriptionPtr->unsubscribe();
    stage->RemovePrim(endToEndTilesetPath);
    tilesetLoadListener.reset();
}

TEST_SUITE("Tileset tests") {
    TEST_CASE("End to end test") {

        // set by the TilesetLoadListener when any tileset successfully loads
        CHECK(endToEndTilesetLoaded);
    }
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
