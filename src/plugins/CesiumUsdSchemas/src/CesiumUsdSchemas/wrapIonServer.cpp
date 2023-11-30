#include ".//ionServer.h"
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
_CreateIonServerUrlAttr(CesiumIonServer &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonServerUrlAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateIonServerApiUrlAttr(CesiumIonServer &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonServerApiUrlAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateIonServerApplicationIdAttr(CesiumIonServer &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonServerApplicationIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int64), writeSparsely);
}
        
static UsdAttribute
_CreateProjectDefaultIonAccessTokenAttr(CesiumIonServer &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateProjectDefaultIonAccessTokenAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateProjectDefaultIonAccessTokenIdAttr(CesiumIonServer &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateProjectDefaultIonAccessTokenIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}

static std::string
_Repr(const CesiumIonServer &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.IonServer(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumIonServer()
{
    typedef CesiumIonServer This;

    class_<This, bases<UsdTyped> >
        cls("IonServer");

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

        
        .def("GetIonServerUrlAttr",
             &This::GetIonServerUrlAttr)
        .def("CreateIonServerUrlAttr",
             &_CreateIonServerUrlAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetIonServerApiUrlAttr",
             &This::GetIonServerApiUrlAttr)
        .def("CreateIonServerApiUrlAttr",
             &_CreateIonServerApiUrlAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetIonServerApplicationIdAttr",
             &This::GetIonServerApplicationIdAttr)
        .def("CreateIonServerApplicationIdAttr",
             &_CreateIonServerApplicationIdAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetProjectDefaultIonAccessTokenAttr",
             &This::GetProjectDefaultIonAccessTokenAttr)
        .def("CreateProjectDefaultIonAccessTokenAttr",
             &_CreateProjectDefaultIonAccessTokenAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetProjectDefaultIonAccessTokenIdAttr",
             &This::GetProjectDefaultIonAccessTokenIdAttr)
        .def("CreateProjectDefaultIonAccessTokenIdAttr",
             &_CreateProjectDefaultIonAccessTokenIdAttr,
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
