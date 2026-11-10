#include "cesium/omniverse/CesiumContext.h"

#include <ovstage/ovstage.h>

CesiumContext::CesiumContext(ovstage_instance_t* pStage)
    : _pStage(pStage) {
    // TODO: temp
    path_dictionary_instance_t* pDict = ovstage_get_path_dictionary(pStage);
    (void)pDict;
}
