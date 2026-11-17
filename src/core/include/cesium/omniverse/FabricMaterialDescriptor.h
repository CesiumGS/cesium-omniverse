#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/sdf/path.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace CesiumGltf {
struct Model;
struct MeshPrimitive;
} // namespace CesiumGltf

namespace cesium::omniverse {

class Context;
enum class FeatureIdType;
enum class OverlayRenderMethod;
struct FeaturesInfo;
struct RasterOverlaysInfo;
struct MaterialInfo;
struct PropertyDescriptor;

/**
* @brief A descriptor used to initialize a {@link FabricMaterial} and {@link FabricMaterialPool}.
*
* The descriptor uniquely identifies the topology of a {@link FabricMaterial} i.e. what Fabric prims
* need to be created and how they're connected. It is distinct from {@MaterialInfo} which
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
        const MaterialInfo& materialInfo,
        const FeaturesInfo& featuresInfo,
        const RasterOverlaysInfo& rasterOverlaysInfo,
        const pxr::SdfPath& tilesetMaterialPath);

    [[nodiscard]] bool hasVertexColors() const;
    [[nodiscard]] bool hasBaseColorTexture() const;
    [[nodiscard]] const std::vector<FeatureIdType>& getFeatureIdTypes() const;
    [[nodiscard]] const std::vector<OverlayRenderMethod>& getOverlayRenderMethods() const;
    [[nodiscard]] bool hasTilesetMaterial() const;
    [[nodiscard]] const pxr::SdfPath& getTilesetMaterialPath() const;
    [[nodiscard]] const std::vector<PropertyDescriptor>& getStyleableProperties() const;
    [[nodiscard]] const std::map<std::string, std::string>& getUnsupportedPropertyWarnings() const;

    bool operator==(const FabricMaterialDescriptor& other) const = default;

  private:
    bool _hasVertexColors;
    bool _hasBaseColorTexture;
    std::vector<FeatureIdType> _featureIdTypes;
    std::vector<OverlayRenderMethod> _overlayRenderMethods;
    pxr::SdfPath _tilesetMaterialPath;
    std::vector<PropertyDescriptor> _styleableProperties;
    std::map<std::string, std::string> _unsupportedPropertyWarnings;
};

} // namespace cesium::omniverse
