#ifndef CESIUM_OMNIVERSE_STAGE_H
#define CESIUM_OMNIVERSE_STAGE_H

#include <ovstage/ovstage.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CesiumContext CesiumContext;

bool cesium_context_create_instance(ovstage_instance_t* stage, CesiumContext** out);
bool cesium_context_destroy_instance(CesiumContext* instance);

#ifdef __cplusplus
}
#endif

#endif // CESIUM_OMNIVERSE_STAGE_H
