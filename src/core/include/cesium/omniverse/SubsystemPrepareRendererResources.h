#pragma once

#ifdef CESIUM_OMNI_MSVC
#pragma push_macro("OPAQUE")
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>

namespace cesium::omniverse {

class Context;
class OmniTileset;

class SubsystemPrepareRendererResources : public Cesium3DTilesSelection::IPrepareRendererResources {
  public:
    SubsystemPrepareRendererResources(Context* pContext, OmniTileset* pTileset);
    // NOLINTNEXTLINE(modernize-use-override)
    virtual ~SubsystemPrepareRendererResources() = default;

    [[nodiscard]] bool tilesetExists() const;
    void detachTileset();

  protected:
    Context* _pContext;
    OmniTileset* _pTileset;
};

} // namespace cesium::omniverse
