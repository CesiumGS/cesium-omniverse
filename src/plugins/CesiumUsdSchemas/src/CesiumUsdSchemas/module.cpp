#include "pxr/pxr.h"
#include "pxr/base/tf/pyModule.h"

PXR_NAMESPACE_USING_DIRECTIVE

TF_WRAP_MODULE
{
  TF_WRAP(CesiumCartographicPolygon);
  TF_WRAP(CesiumData);
  TF_WRAP(CesiumGeoreference);
  TF_WRAP(CesiumGlobeAnchorAPI);
  TF_WRAP(CesiumIonServer);
  TF_WRAP(CesiumImagery);
  TF_WRAP(CesiumIonImagery);
  TF_WRAP(CesiumPolygonImagery);
  TF_WRAP(CesiumSession);
  TF_WRAP(CesiumTileset);
  TF_WRAP(CesiumTokens);
}
