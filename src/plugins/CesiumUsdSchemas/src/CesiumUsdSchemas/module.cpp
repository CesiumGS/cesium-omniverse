#include "Python.h"
#include "pxr/pxr.h"
#include "pxr/base/tf/pyModule.h"

PXR_NAMESPACE_USING_DIRECTIVE

TF_WRAP_MODULE
{
  TF_WRAP(CesiumData);
  TF_WRAP(CesiumImagery);
  TF_WRAP(CesiumTilesetAPI);
  TF_WRAP(CesiumTokens);
}