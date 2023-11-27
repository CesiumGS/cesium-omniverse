#include ".//imagery.h"
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
_CreateShowCreditsOnScreenAttr(CesiumImagery &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateShowCreditsOnScreenAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateAlphaAttr(CesiumImagery &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateAlphaAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float), writeSparsely);
}
        
static UsdAttribute
_CreateIonAssetIdAttr(CesiumImagery &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonAssetIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Int64), writeSparsely);
}
        
static UsdAttribute
_CreateIonAccessTokenAttr(CesiumImagery &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateIonAccessTokenAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}

static std::string
_Repr(const CesiumImagery &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "CesiumUsdSchemas.Imagery(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumImagery()
{
    typedef CesiumImagery This;

    class_<This, bases<UsdTyped> >
        cls("Imagery");

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
        
        .def("GetIonAssetIdAttr",
             &This::GetIonAssetIdAttr)
        .def("CreateIonAssetIdAttr",
             &_CreateIonAssetIdAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetIonAccessTokenAttr",
             &This::GetIonAccessTokenAttr)
        .def("CreateIonAccessTokenAttr",
             &_CreateIonAccessTokenAttr,
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
