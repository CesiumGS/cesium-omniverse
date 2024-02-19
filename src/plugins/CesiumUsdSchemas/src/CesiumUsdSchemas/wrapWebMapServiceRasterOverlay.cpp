#include ".//webMapServiceRasterOverlay.h"
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
_CreateBaseUrlAttr(CesiumWebMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateBaseUrlAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateLayersAttr(CesiumWebMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateLayersAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateTileWidthAttr(CesiumWebMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateTileWidthAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateTileHeightAttr(CesiumWebMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateTileHeightAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateMinimumLevelAttr(CesiumWebMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMinimumLevelAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateMaximumLevelAttr(CesiumWebMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMaximumLevelAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}

static std::string
_Repr(const CesiumWebMapServiceRasterOverlay &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.WebMapServiceRasterOverlay(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumWebMapServiceRasterOverlay()
{
    typedef CesiumWebMapServiceRasterOverlay This;

    class_<This, bases<CesiumRasterOverlay> >
        cls("WebMapServiceRasterOverlay");

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

        
        .def("GetBaseUrlAttr",
             &This::GetBaseUrlAttr)
        .def("CreateBaseUrlAttr",
             &_CreateBaseUrlAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetLayersAttr",
             &This::GetLayersAttr)
        .def("CreateLayersAttr",
             &_CreateLayersAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetTileWidthAttr",
             &This::GetTileWidthAttr)
        .def("CreateTileWidthAttr",
             &_CreateTileWidthAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetTileHeightAttr",
             &This::GetTileHeightAttr)
        .def("CreateTileHeightAttr",
             &_CreateTileHeightAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMinimumLevelAttr",
             &This::GetMinimumLevelAttr)
        .def("CreateMinimumLevelAttr",
             &_CreateMinimumLevelAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMaximumLevelAttr",
             &This::GetMaximumLevelAttr)
        .def("CreateMaximumLevelAttr",
             &_CreateMaximumLevelAttr,
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
