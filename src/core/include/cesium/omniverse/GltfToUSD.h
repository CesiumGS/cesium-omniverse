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

namespace cesium::omniverse {
struct GltfToUSD {
    static std::filesystem::path CesiumMemLocation;
    static pxr::UsdPrim convertToUSD(
        pxr::UsdStageRefPtr& stage,
        const pxr::SdfPath& modelPath,
        const CesiumGltf::Model& model,
        const glm::dmat4& transform);

    static std::vector<std::byte> writeImageToBmp(const CesiumGltf::ImageCesium& img);
    static void insertRasterOverlayTexture(
        const pxr::UsdPrim& parent,
        std::vector<std::byte>&& image,
        const glm::dvec2& translation,
        const glm::dvec2& scale);

    static void removeRasterOverlayTexture(const pxr::UsdPrim& parent);
};
} // namespace cesium::omniverse
