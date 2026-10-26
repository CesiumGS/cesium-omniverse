#pragma once

#include "FabricMesh.h"

#include <glm/glm.hpp>

#include <vector>

namespace cesium::omniverse {

struct FabricRenderResources {
    FabricRenderResources(std::vector<FabricMesh>&& fabricMeshes_)
        : fabricMeshes(std::move(fabricMeshes_)) {}
    ~FabricRenderResources() = default;
    FabricRenderResources(const FabricRenderResources&) = delete;
    FabricRenderResources& operator=(const FabricRenderResources&) = delete;
    FabricRenderResources(FabricRenderResources&&) noexcept = default;
    FabricRenderResources& operator=(FabricRenderResources&&) noexcept = default;

    std::vector<FabricMesh> fabricMeshes;
};

} // namespace cesium::omniverse
