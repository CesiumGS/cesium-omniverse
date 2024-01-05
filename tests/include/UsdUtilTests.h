#pragma once
#include <pxr/usd/usd/common.h>

namespace cesium::omniverse {
class Context;
}

void setUpUsdUtilTests(cesium::omniverse::Context* pContext, const PXR_NS::SdfPath& rootPath);
void cleanUpUsdUtilTests(const PXR_NS::UsdStageRefPtr& stage);
