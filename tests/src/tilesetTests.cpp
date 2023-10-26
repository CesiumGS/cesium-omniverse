#include "tilesetTests.h"

#include "testUtils.h"

#include "cesium/omniverse/UsdUtil.h"

pxr::SdfPath endToEndTilesetPath;

using namespace cesium::omniverse;

void setUpTilesetTests(const pxr::SdfPath& rootPath) {

    endToEndTilesetPath = rootPath.AppendChild(pxr::TfToken("endToEndTileset"));

    auto foo = UsdUtil::defineCesiumTileset(endToEndTilesetPath);

    std::string tilesetFilePath = "file://" TEST_WORKING_DIRECTORY "/tests/testAssets/tilesets/agi/tileset.json";

    foo.GetSourceTypeAttr().Set(pxr::TfToken("url"));
    foo.GetUrlAttr().Set(tilesetFilePath);
}
void cleanUpTilesetTests(const pxr::UsdStageRefPtr& stage) {

    stage->RemovePrim(endToEndTilesetPath);
}
