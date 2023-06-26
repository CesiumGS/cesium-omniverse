#include ".//georeference.h"
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
_CreateGeoreferenceOriginLongitudeAttr(CesiumGeoreference &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateGeoreferenceOriginLongitudeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}
        
static UsdAttribute
_CreateGeoreferenceOriginLatitudeAttr(CesiumGeoreference &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateGeoreferenceOriginLatitudeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}
        
static UsdAttribute
_CreateGeoreferenceOriginHeightAttr(CesiumGeoreference &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateGeoreferenceOriginHeightAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Double), writeSparsely);
}

static std::string
_Repr(const CesiumGeoreference &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.Georeference(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumGeoreference()
{
    typedef CesiumGeoreference This;

    class_<This, bases<UsdTyped> >
        cls("Georeference");

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

        
        .def("GetGeoreferenceOriginLongitudeAttr",
             &This::GetGeoreferenceOriginLongitudeAttr)
        .def("CreateGeoreferenceOriginLongitudeAttr",
             &_CreateGeoreferenceOriginLongitudeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetGeoreferenceOriginLatitudeAttr",
             &This::GetGeoreferenceOriginLatitudeAttr)
        .def("CreateGeoreferenceOriginLatitudeAttr",
             &_CreateGeoreferenceOriginLatitudeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetGeoreferenceOriginHeightAttr",
             &This::GetGeoreferenceOriginHeightAttr)
        .def("CreateGeoreferenceOriginHeightAttr",
             &_CreateGeoreferenceOriginHeightAttr,
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
