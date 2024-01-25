#pragma once
#include <pxr/usd/usd/common.h>

namespace cesium::omniverse {
class Context;
}

void setUpUsdUtilTests(cesium::omniverse::Context* pContext, const pxr::SdfPath& rootPath);
void cleanUpUsdUtilTests(const pxr::UsdStageRefPtr& stage);
