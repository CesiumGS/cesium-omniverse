#pragma once
#include <pxr/usd/usd/common.h>

namespace cesium::omniverse {
class Context;
}

void setUpTilesetTests(cesium::omniverse::Context* pContext, const PXR_NS::SdfPath& rootPath);
void cleanUpTilesetTests(const PXR_NS::UsdStageRefPtr& stage);
