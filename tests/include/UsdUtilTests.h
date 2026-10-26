#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/usd/common.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace cesium::omniverse {
class Context;
}

void setUpUsdUtilTests(cesium::omniverse::Context* pContext, const pxr::SdfPath& rootPath);
void cleanUpUsdUtilTests(const pxr::UsdStageRefPtr& stage);
