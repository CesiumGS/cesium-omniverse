#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <glm/glm.hpp>

#ifdef CESIUM_OMNI_GCC
#define _GLIBCXX_PERMIT_BACKWARD_HASH
#endif

#include <pxr/usd/usd/prim.h>

#include <string>

namespace Cesium {
struct GltfToUSD {
    static pxr::UsdPrim convertToUSD(
        pxr::UsdStageRefPtr& stage,
        const pxr::SdfPath& modelPath,
        const CesiumGltf::Model& model,
        const glm::dmat4& transform);
};
} // namespace Cesium
