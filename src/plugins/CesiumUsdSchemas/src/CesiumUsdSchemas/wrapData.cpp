#include ".//data.h"
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
_CreateProjectDefaultIonAccessTokenAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateProjectDefaultIonAccessTokenAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateProjectDefaultIonAccessTokenIdAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateProjectDefaultIonAccessTokenIdAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->String), writeSparsely);
}
        
static UsdAttribute
_CreateDebugDisableMaterialsAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugDisableMaterialsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateDebugDisableTexturesAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugDisableTexturesAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateDebugDisableGeometryPoolAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugDisableGeometryPoolAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateDebugDisableMaterialPoolAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugDisableMaterialPoolAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreateDebugGeometryPoolInitialCapacityAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugGeometryPoolInitialCapacityAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt64), writeSparsely);
}
        
static UsdAttribute
_CreateDebugMaterialPoolInitialCapacityAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugMaterialPoolInitialCapacityAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt64), writeSparsely);
}
        
static UsdAttribute
_CreateDebugRandomColorsAttr(CesiumData &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreateDebugRandomColorsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}

static std::string
_Repr(const CesiumData &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "Cesium.Data(%s)",
        primRepr.c_str());
}

} // anonymous namespace

void wrapCesiumData()
{
    typedef CesiumData This;

    class_<This, bases<UsdTyped> >
        cls("Data");

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
        
        .def("GetDebugDisableMaterialsAttr",
             &This::GetDebugDisableMaterialsAttr)
        .def("CreateDebugDisableMaterialsAttr",
             &_CreateDebugDisableMaterialsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetDebugDisableTexturesAttr",
             &This::GetDebugDisableTexturesAttr)
        .def("CreateDebugDisableTexturesAttr",
             &_CreateDebugDisableTexturesAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetDebugDisableGeometryPoolAttr",
             &This::GetDebugDisableGeometryPoolAttr)
        .def("CreateDebugDisableGeometryPoolAttr",
             &_CreateDebugDisableGeometryPoolAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetDebugDisableMaterialPoolAttr",
             &This::GetDebugDisableMaterialPoolAttr)
        .def("CreateDebugDisableMaterialPoolAttr",
             &_CreateDebugDisableMaterialPoolAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetDebugGeometryPoolInitialCapacityAttr",
             &This::GetDebugGeometryPoolInitialCapacityAttr)
        .def("CreateDebugGeometryPoolInitialCapacityAttr",
             &_CreateDebugGeometryPoolInitialCapacityAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetDebugMaterialPoolInitialCapacityAttr",
             &This::GetDebugMaterialPoolInitialCapacityAttr)
        .def("CreateDebugMaterialPoolInitialCapacityAttr",
             &_CreateDebugMaterialPoolInitialCapacityAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetDebugRandomColorsAttr",
             &This::GetDebugRandomColorsAttr)
        .def("CreateDebugRandomColorsAttr",
             &_CreateDebugRandomColorsAttr,
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
