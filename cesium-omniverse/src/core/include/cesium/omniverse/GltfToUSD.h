#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <CesiumGltf/Model.h>
#include <glm/glm.hpp>
#include <pxr/usd/usd/prim.h>

#include <filesystem>
#include <string>

namespace Cesium {
struct GltfToUSD {
    static std::filesystem::path CesiumMemLocation;
    static pxr::UsdPrim convertToUSD(
        pxr::UsdStageRefPtr& stage,
        const pxr::SdfPath& modelPath,
        const CesiumGltf::Model& model,
        const glm::dmat4& transform);
};
} // namespace Cesium
