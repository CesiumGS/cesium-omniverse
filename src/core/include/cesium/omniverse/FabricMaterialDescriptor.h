#pragma once

#include <pxr/usd/sdf/path.h>

namespace CesiumGltf {
struct Model;
struct MeshPrimitive;
} // namespace CesiumGltf

namespace cesium::omniverse {

class Context;
enum class FabricFeatureIdType;
enum class FabricOverlayRenderMethod;
struct FabricFeaturesInfo;
struct FabricRasterOverlayLayersInfo;
struct FabricMaterialInfo;
struct FabricPropertyDescriptor;

/**
* @brief A descriptor used to initialize a {@link FabricMaterial} and {@link FabricMaterialPool}.
*
* The descriptor uniquely identifies the topology of a {@link FabricMaterial} i.e. what Fabric prims
* need to be created and how they're connected. It is distinct from {@FabricMaterialInfo} which
* supplies the actual material values.
*
* Materials that have the same material descriptor will be assigned to the same material pool.
* To reduce the number of material pools that are needed the list of member variables should be
* as limited as possible.
*/
class FabricMaterialDescriptor {
  public:
    FabricMaterialDescriptor(
        const Context& context,
        const CesiumGltf::Model& model,
        const CesiumGltf::MeshPrimitive& primitive,
        const FabricMaterialInfo& materialInfo,
        const FabricFeaturesInfo& featuresInfo,
        const FabricRasterOverlayLayersInfo& rasterOverlayLayersInfo,
        const pxr::SdfPath& tilesetMaterialPath);

    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] const std::vector<FabricFeatureIdType>& getFeatureIdTypes() const;
    [[nodiscard]] const std::vector<FabricOverlayRenderMethod>& getRasterOverlayRenderMethods() const;
    [[nodiscard]] bool hasTilesetMaterial() const;
    [[nodiscard]] const pxr::SdfPath& getTilesetMaterialPath() const;
    [[nodiscard]] const std::vector<FabricPropertyDescriptor>& getStyleableProperties() const;

    bool operator==(const FabricMaterialDescriptor& other) const;

  private:
    bool _hasVertexColors;
    bool _hasBaseColorTexture;
    std::vector<FabricFeatureIdType> _featureIdTypes;
    std::vector<FabricOverlayRenderMethod> _rasterOverlayRenderMethods;
    pxr::SdfPath _tilesetMaterialPath;
    std::vector<FabricPropertyDescriptor> _styleableProperties;
};

} // namespace cesium::omniverse
