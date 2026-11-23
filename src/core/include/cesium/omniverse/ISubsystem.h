#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(push)
#pragma warning(disable : 4244 4305 4100 4201 4127 4251)
#endif

#include <pxr/usd/usd/common.h>

#ifdef CESIUM_OMNI_MSVC
#pragma warning(pop)
#endif

namespace Cesium3DTilesSelection {
class Tileset;
}

namespace cesium::omniverse {

class Context;
class IBroadcast;
class ILogger;
class ISettings;
class OmniTileset;
class SubsystemPrepareRendererResources;
struct DebugSettings;
struct SubsystemStatistics;

class ISubsystem {
  public:
    virtual ~ISubsystem() = 0;

    virtual void clear() = 0;
    virtual void applyDebugSettings(const DebugSettings& debugSettings) = 0;
    virtual void releaseStage() = 0;
    virtual void acquireStage(int64_t usdStageId) = 0;
    [[nodiscard]] virtual SubsystemStatistics getStatistics() const = 0;
    virtual void setTilesetTransform(int64_t tilesetId, const glm::dmat4& ecefToPrimWorldTransform) = 0;
    [[nodiscard]] virtual std::string printStage() const = 0;
    virtual std::shared_ptr<SubsystemPrepareRendererResources>
    createPrepareRendererResources(Context* pContext, OmniTileset* pTileset) = 0;
    [[nodiscard]] virtual bool materialHasCesiumNodes(const pxr::SdfPath& materialPath) const = 0;
    [[nodiscard]] virtual bool
    isShaderConnectedToMaterial(const pxr::SdfPath& materialPath, const pxr::SdfPath& shaderPath) const = 0;
    virtual void updateShaderInput(
        const pxr::SdfPath& materialPath,
        const pxr::SdfPath& shaderPath,
        const pxr::TfToken& property) const = 0;
    virtual void updateRasterOverlayAlpha(
        const Cesium3DTilesSelection::Tileset& tileset,
        uint64_t rasterOverlayIndex,
        double alpha) const = 0;
    virtual void updateDisplayColorAndOpacity(
        const Cesium3DTilesSelection::Tileset& tileset,
        const glm::dvec3& displayColor,
        double displayOpacity) const = 0;
    virtual void updateShaderInput(
        const Cesium3DTilesSelection::Tileset& tileset,
        const pxr::SdfPath& shaderPath,
        const pxr::TfToken& attributeName) const = 0;
    virtual void setContentVisiblity(void* pRenderResources, bool visible) const = 0;
    [[nodiscard]] virtual std::shared_ptr<ILogger> getLogger() const = 0;
    [[nodiscard]] virtual const ISettings& getSettings() const = 0;
    [[nodiscard]] virtual const IBroadcast& getBroadcast() const = 0;
};

} // namespace cesium::omniverse
