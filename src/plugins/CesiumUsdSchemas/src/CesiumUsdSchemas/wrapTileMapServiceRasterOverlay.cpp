#include ".//tileMapServiceRasterOverlay.h"
#include "pxr/usd/usd/schemaBase.h"

#include "pxr/usd/sdf/primSpec.h"

#include "pxr/usd/usd/pyConversions.h"
#include "pxr/base/tf/pyContainerConversions.h"
#include "pxr/base/tf/pyResultConversions.h"
#include "pxr/base/tf/pyUtils.h"
#include "pxr/base/tf/wrapTypeHelpers.h"

#include "pxr/external/boost/python.hpp"

#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace pxr_boost::python;

namespace {

#define WRAP_CUSTOM                                                     \
    template <class Cls> static void _CustomWrapCode(Cls &_class)

// fwd decl.
WRAP_CUSTOM;

        
static UsdAttribute
_CreateUrlAttr(CesiumTileMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateUrlAttr(
        UsdPythonToSdfType(TfPyObjWrapper(defaultVal), SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateSpecifyZoomLevelsAttr(CesiumTileMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateSpecifyZoomLevelsAttr(
        UsdPythonToSdfType(TfPyObjWrapper(defaultVal), SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateMinimumZoomLevelAttr(CesiumTileMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMinimumZoomLevelAttr(
        UsdPythonToSdfType(TfPyObjWrapper(defaultVal), SdfValueTypeNames->Int), writeSparsely);
}
        
static UsdAttribute
_CreateMaximumZoomLevelAttr(CesiumTileMapServiceRasterOverlay &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateMaximumZoomLevelAttr(
        UsdPythonToSdfType(TfPyObjWrapper(defaultVal), SdfValueTypeNames->Int), writeSparsely);
}

static std::string
_Repr(const CesiumTileMapServiceRasterOverlay &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.TileMapServiceRasterOverlay(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumTileMapServiceRasterOverlay()
{
    typedef CesiumTileMapServiceRasterOverlay This;

    class_<This, bases<CesiumRasterOverlay> >
        cls("TileMapServiceRasterOverlay");

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
