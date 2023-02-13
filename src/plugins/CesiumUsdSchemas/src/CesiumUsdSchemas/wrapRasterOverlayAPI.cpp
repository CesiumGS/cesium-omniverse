#include ".//rasterOverlayAPI.h"
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
_CreateRasterOverlayIdAttr(CesiumRasterOverlayAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateRasterOverlayIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateNameAttr(CesiumRasterOverlayAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateNameAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateIonTokenIdAttr(CesiumRasterOverlayAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonTokenIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateIonTokenAttr(CesiumRasterOverlayAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonTokenAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}

static std::string
_Repr(const CesiumRasterOverlayAPI &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "Cesium.RasterOverlayAPI(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumRasterOverlayAPI()
{
    typedef CesiumRasterOverlayAPI This;

    class_<This, bases<UsdAPISchemaBase> >
        cls("RasterOverlayAPI");

    cls
        .def(init<UsdPrim>(arg("prim")))
        .def(init<UsdSchemaBase const&>(arg("schemaObj")))
        .def(TfTypePythonClass())

        .def("Get", &This::Get, (arg("stage"), arg("path")))
        .staticmethod("Get")

        .def("Apply", &This::Apply, (arg("prim")))
        .staticmethod("Apply")

        .def("GetSchemaAttributeNames",
             &This::GetSchemaAttributeNames,
             arg("includeInherited")=true,
             return_value_policy<TfPySequenceToList>())
        .staticmethod("GetSchemaAttributeNames")

        .def("_GetStaticTfType", (TfType const &(*)()) TfType::Find<This>,
             return_value_policy<return_by_value>())
        .staticmethod("_GetStaticTfType")

        .def(!self)

        
        .def("GetRasterOverlayIdAttr",
             &This::GetRasterOverlayIdAttr)
        .def("CreateRasterOverlayIdAttr",
             &_CreateRasterOverlayIdAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetNameAttr",
             &This::GetNameAttr)
        .def("CreateNameAttr",
             &_CreateNameAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetIonTokenIdAttr",
             &This::GetIonTokenIdAttr)
        .def("CreateIonTokenIdAttr",
             &_CreateIonTokenIdAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetIonTokenAttr",
             &This::GetIonTokenAttr)
        .def("CreateIonTokenAttr",
             &_CreateIonTokenAttr,
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