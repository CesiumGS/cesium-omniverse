#include "cesium/omniverse/CesiumContext.h"
#include "cesium/omniverse/CesiumOmniverseOvstage.h"

#include <cassert>

struct ovstage_instance_t;

extern "C" {

bool cesium_context_create_instance(ovstage_instance_t* stage, CesiumContext** out) {
    assert(stage);

    *out = new CesiumContext(stage);
    return true;
}

bool cesium_context_destroy_instance(CesiumContext* instance) {
    delete instance;
    return true;
}

} // extern "C"
