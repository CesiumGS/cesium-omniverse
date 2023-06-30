#pragma once

#include <cstdint>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

class FabricGeometryDefinition {
  public:
    FabricGeometryDefinition(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        bool smoothNormals,
        bool hasImagery,
        bool disableMaterials);

    [[nodiscard]] bool hasMaterial() const;
    [[nodiscard]] bool hasTexcoords() const;
    [[nodiscard]] bool hasNormals() const;
    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool getDoubleSided() const;

    bool operator==(const FabricGeometryDefinition& other) const;

  private:
    bool _hasMaterial{false};
    bool _hasTexcoords{false};
    bool _hasNormals{false};
    bool _hasVertexColors{false};
    bool _doubleSided{false};
};

} // namespace cesium::omniverse
