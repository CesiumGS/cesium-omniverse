#include ".//webMapTileServiceRasterOverlay.h"
#include "pxr/usd/usd/schemaBase.h"

#include "pxr/usd/sdf/primSpec.h"

#include "pxr/usd/usd/pyConversions.h"
#include "pxr/base/tf/pyContainerConversions.h"
#include "pxr/base/tf/pyResultConversions.h"
#include "pxr/base/tf/pyUtils.h"
#include "pxr/base/tf/wrapTypeHelpers.h"

#include <boost/python.hpp>

#include <string>

using namespace boost::python;

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

#define WRAP_CUSTOM                                                     \
    template <class Cls> static void _CustomWrapCode(Cls &_class)

// fwd decl.
WRAP_CUSTOM;

        
static UsdAttribute
_CreateUrlAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateUrlAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateLayerAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateLayerAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateStyleAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateStyleAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateFormatAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateFormatAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateTileMatrixSetIdAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateTileMatrixSetIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateSpecifyTileMatrixSetLabelsAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateSpecifyTileMatrixSetLabelsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateTileMatrixSetLabelPrefixAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateTileMatrixSetLabelPrefixAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateTileMatrixSetLabelsAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateTileMatrixSetLabelsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateUseWebMercatorProjectionAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateUseWebMercatorProjectionAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateSpecifyTilingSchemeAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateSpecifyTilingSchemeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateRootTilesXAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateRootTilesXAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateRootTilesYAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateRootTilesYAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateWestAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateWestAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}
        
static UsdAttribute
_CreateEastAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateEastAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}
        
static UsdAttribute
_CreateSouthAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateSouthAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}
        
static UsdAttribute
_CreateNorthAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateNorthAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}
        
static UsdAttribute
_CreateSpecifyZoomLevelsAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateSpecifyZoomLevelsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateMinimumZoomLevelAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMinimumZoomLevelAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateMaximumZoomLevelAttr(CesiumWebMapTileServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMaximumZoomLevelAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}

static std::string
_Repr(const CesiumWebMapTileServiceRasterOverlay &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.WebMapTileServiceRasterOverlay(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumWebMapTileServiceRasterOverlay()
{
    typedef CesiumWebMapTileServiceRasterOverlay This;

    class_<This, bases<CesiumRasterOverlay> >
        cls("WebMapTileServiceRasterOverlay");

    cls
        .def(init<UsdPrim>(arg("prim")))
        .def(init<UsdSchemaBase const&>(arg("schemaObj")))
        .def(TfTypePythonClass())

        .def("Get", &This::Get, (arg("stage"), arg("path")))
        .staticmethod("Get")

        .def("Define", &This::Define, (arg("stage"), arg("path")))
        .staticmethod("Define")

        .def("GetSchemaAttributeNames",
             &This::GetSchemaAttributeNames,
             arg("includeInherited")=true,
             return_value_policy<TfPySequenceToList>())
        .staticmethod("GetSchemaAttributeNames")

        .def("_GetStaticTfType", (TfType const &(*)()) TfType::Find<This>,
             return_value_policy<return_by_value>())
        .staticmethod("_GetStaticTfType")

        .def(!self)

        
        .def("GetUrlAttr",
             &This::GetUrlAttr)
        .def("CreateUrlAttr",
             &_CreateUrlAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetLayerAttr",
             &This::GetLayerAttr)
        .def("CreateLayerAttr",
             &_CreateLayerAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetStyleAttr",
             &This::GetStyleAttr)
        .def("CreateStyleAttr",
             &_CreateStyleAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetFormatAttr",
             &This::GetFormatAttr)
        .def("CreateFormatAttr",
             &_CreateFormatAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetTileMatrixSetIdAttr",
             &This::GetTileMatrixSetIdAttr)
        .def("CreateTileMatrixSetIdAttr",
             &_CreateTileMatrixSetIdAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetSpecifyTileMatrixSetLabelsAttr",
             &This::GetSpecifyTileMatrixSetLabelsAttr)
        .def("CreateSpecifyTileMatrixSetLabelsAttr",
             &_CreateSpecifyTileMatrixSetLabelsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetTileMatrixSetLabelPrefixAttr",
             &This::GetTileMatrixSetLabelPrefixAttr)
        .def("CreateTileMatrixSetLabelPrefixAttr",
             &_CreateTileMatrixSetLabelPrefixAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetTileMatrixSetLabelsAttr",
             &This::GetTileMatrixSetLabelsAttr)
        .def("CreateTileMatrixSetLabelsAttr",
             &_CreateTileMatrixSetLabelsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetUseWebMercatorProjectionAttr",
             &This::GetUseWebMercatorProjectionAttr)
        .def("CreateUseWebMercatorProjectionAttr",
             &_CreateUseWebMercatorProjectionAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetSpecifyTilingSchemeAttr",
             &This::GetSpecifyTilingSchemeAttr)
        .def("CreateSpecifyTilingSchemeAttr",
             &_CreateSpecifyTilingSchemeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetRootTilesXAttr",
             &This::GetRootTilesXAttr)
        .def("CreateRootTilesXAttr",
             &_CreateRootTilesXAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetRootTilesYAttr",
             &This::GetRootTilesYAttr)
        .def("CreateRootTilesYAttr",
             &_CreateRootTilesYAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetWestAttr",
             &This::GetWestAttr)
        .def("CreateWestAttr",
             &_CreateWestAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetEastAttr",
             &This::GetEastAttr)
        .def("CreateEastAttr",
             &_CreateEastAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetSouthAttr",
             &This::GetSouthAttr)
        .def("CreateSouthAttr",
             &_CreateSouthAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetNorthAttr",
             &This::GetNorthAttr)
        .def("CreateNorthAttr",
             &_CreateNorthAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetSpecifyZoomLevelsAttr",
             &This::GetSpecifyZoomLevelsAttr)
        .def("CreateSpecifyZoomLevelsAttr",
             &_CreateSpecifyZoomLevelsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMinimumZoomLevelAttr",
             &This::GetMinimumZoomLevelAttr)
        .def("CreateMinimumZoomLevelAttr",
             &_CreateMinimumZoomLevelAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMaximumZoomLevelAttr",
             &This::GetMaximumZoomLevelAttr)
        .def("CreateMaximumZoomLevelAttr",
             &_CreateMaximumZoomLevelAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))

        .def("__repr__", ::_Repr)
    ;

    _CustomWrapCode(cls);
}

// ===================================================================== //
// Feel free to add custom code below this line, it will be preserved by 
// the code generator.  The entry point for your custom code should look
// minimally like the following:
//
// WRAP_CUSTOM {
//     _class
//         .def("MyCustomMethod", ...)
//     ;
// }
//
// Of course any other ancillary or support code may be provided.
// 
// Just remember to wrap code in the appropriate delimiters:
// 'namespace {', '}'.
//
// ===================================================================== //
// --(BEGIN CUSTOM CODE)--

namespace {

WRAP_CUSTOM {
}

}
