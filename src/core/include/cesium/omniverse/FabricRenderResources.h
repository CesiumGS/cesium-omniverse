#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace cesium::omniverse {

struct FabricMesh;

struct FabricRenderResources {
    FabricRenderResources() = default;
    ~FabricRenderResources() = default;
    FabricRenderResources(const FabricRenderResources&) = delete;
    FabricRenderResources& operator=(const FabricRenderResources&) = delete;
    FabricRenderResources(FabricRenderResources&&) noexcept = default;
    FabricRenderResources& operator=(FabricRenderResources&&) noexcept = default;

    std::vector<FabricMesh> fabricMeshes;
};

} // namespace cesium::omniverse
