#pragma once

#include <cstdint>
#include <set>

namespace CesiumGltf {
struct MeshPrimitive;
struct Model;
} // namespace CesiumGltf

namespace cesium::omniverse {

struct FabricFeaturesInfo;
struct FabricVertexAttributeDescriptor;

/**
* @brief A descriptor used to initialize a {@link FabricGeometry} and {@link FabricGeometryPool}.
*
* The descriptor uniquely identifies the topology of a {@link FabricGeometry} i.e. what Fabric
* attributes are added to the prim, but not the actual values.
*
* Geometries that have the same geometry descriptor will be assigned to the same geometry pool.
* To reduce the number of geometry pools that are needed the list of member variables should be
* as limited as possible.
*/
class FabricGeometryDescriptor {
  public:
    FabricGeometryDescriptor(
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const FabricFeaturesInfo& featuresInfo,
        bool smoothNormals);

    [[nodiscard]] bool hasNormals() const;
    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool hasVertexIds() const;
    [[nodiscard]] uint64_t getTexcoordSetCount() const;
    [[nodiscard]] const std::set<FabricVertexAttributeDescriptor>& getCustomVertexAttributes() const;

    bool operator==(const FabricGeometryDescriptor& other) const;

  private:
    bool _hasNormals{false};
    bool _hasVertexColors{false};
    bool _hasVertexIds{false};
    uint64_t _texcoordSetCount{0};

    // std::set is sorted which is important for checking FabricGeometryDescriptor equality
    // Note that features ids are treated as custom vertex attributes since they don't have specific parsing behavior
    std::set<FabricVertexAttributeDescriptor> _customVertexAttributes;
};

} // namespace cesium::omniverse
