#include ".//rasterOverlay.h"
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
_CreateShowCreditsOnScreenAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateShowCreditsOnScreenAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateAlphaAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateAlphaAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float), writeSparsely);
}
        
static UsdAttribute
_CreateOverlayRenderMethodAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateOverlayRenderMethodAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Token), writeSparsely);
}
        
static UsdAttribute
_CreateMaximumScreenSpaceErrorAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMaximumScreenSpaceErrorAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float), writeSparsely);
}
        
static UsdAttribute
_CreateMaximumTextureSizeAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMaximumTextureSizeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateMaximumSimultaneousTileLoadsAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMaximumSimultaneousTileLoadsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateSubTileCacheBytesAttr(CesiumRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateSubTileCacheBytesAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int), writeSparsely);
}

static std::string
_Repr(const CesiumRasterOverlay &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.RasterOverlay(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumRasterOverlay()
{
    typedef CesiumRasterOverlay This;

    class_<This, bases<UsdTyped> >
        cls("RasterOverlay");

    cls
        .def(init<UsdPrim>(arg("prim")))
        .def(init<UsdSchemaBase const&>(arg("schemaObj")))
        .def(TfTypePythonClass())

        .def("Get", &This::Get, (arg("stage"), arg("path")))
        .staticmethod("Get")

        .def("GetSchemaAttributeNames",
             &This::GetSchemaAttributeNames,
             arg("includeInherited")=true,
             return_value_policy<TfPySequenceToList>())
        .staticmethod("GetSchemaAttributeNames")

        .def("_GetStaticTfType", (TfType const &(*)()) TfType::Find<This>,
             return_value_policy<return_by_value>())
        .staticmethod("_GetStaticTfType")

        .def(!self)

        
        .def("GetShowCreditsOnScreenAttr",
             &This::GetShowCreditsOnScreenAttr)
        .def("CreateShowCreditsOnScreenAttr",
             &_CreateShowCreditsOnScreenAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetAlphaAttr",
             &This::GetAlphaAttr)
        .def("CreateAlphaAttr",
             &_CreateAlphaAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetOverlayRenderMethodAttr",
             &This::GetOverlayRenderMethodAttr)
        .def("CreateOverlayRenderMethodAttr",
             &_CreateOverlayRenderMethodAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMaximumScreenSpaceErrorAttr",
             &This::GetMaximumScreenSpaceErrorAttr)
        .def("CreateMaximumScreenSpaceErrorAttr",
             &_CreateMaximumScreenSpaceErrorAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMaximumTextureSizeAttr",
             &This::GetMaximumTextureSizeAttr)
        .def("CreateMaximumTextureSizeAttr",
             &_CreateMaximumTextureSizeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetMaximumSimultaneousTileLoadsAttr",
             &This::GetMaximumSimultaneousTileLoadsAttr)
        .def("CreateMaximumSimultaneousTileLoadsAttr",
             &_CreateMaximumSimultaneousTileLoadsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetSubTileCacheBytesAttr",
             &This::GetSubTileCacheBytesAttr)
        .def("CreateSubTileCacheBytesAttr",
             &_CreateSubTileCacheBytesAttr,
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
