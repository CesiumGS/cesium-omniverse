#pragma once
#include <pxr/usd/usd/common.h>

namespace cesium::omniverse {
class Context;
}

void setUpTilesetTests(cesium::omniverse::Context* pContext, const pxr::SdfPath& rootPath);
void cleanUpTilesetTests(const pxr::UsdStageRefPtr& stage);
