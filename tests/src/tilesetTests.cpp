#include "tilesetTests.h"

#include "cesium/omniverse/UsdUtil.h"

pxr::SdfPath endToEndTilesetPath;

using namespace cesium::omniverse;

void setUpTilesetTests(const pxr::SdfPath& rootPath) {

    endToEndTilesetPath = rootPath.AppendChild(pxr::TfToken("endToEndTileset"));

    UsdUtil::defineCesiumTileset(endToEndTilesetPath);
}
void cleanUpTilesetTests(const pxr::UsdStageRefPtr& stage) {

    stage->RemovePrim(endToEndTilesetPath);
}
