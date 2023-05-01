#pragma once

#include "cesium/omniverse/FabricGeometry.h"
#include "cesium/omniverse/FabricGeometryDefinition.h"
#include "cesium/omniverse/FabricMaterial.h"
#include "cesium/omniverse/FabricMaterialDefinition.h"

#include <memory>

namespace cesium::omniverse {

class FabricGeometryDefinition;
class FabricMaterialDefinition;

class FabricMesh {
  public:
    FabricMesh(std::shared_ptr<FabricGeometry> geometry, std::shared_ptr<FabricMaterial> material);

    std::shared_ptr<FabricGeometry> getGeometry() const;
    std::shared_ptr<FabricMaterial> getMaterial() const;

    void setVisibility(bool visible) const;

  private:
    std::shared_ptr<FabricGeometry> _geometry;
    std::shared_ptr<FabricMaterial> _material;
};

} // namespace cesium::omniverse
