#include ".//ionServer.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumIonServer,
        TfType::Bases< UsdTyped > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumIonServerPrim")
    // to find TfType<CesiumIonServer>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumIonServer>("CesiumIonServerPrim");
}

/* virtual */
CesiumIonServer::~CesiumIonServer()
{
}

/* static */
CesiumIonServer
CesiumIonServer::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumIonServer();
    }
    return CesiumIonServer(stage->GetPrimAtPath(path));
}

/* static */
CesiumIonServer
CesiumIonServer::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumIonServerPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumIonServer();
    }
    return CesiumIonServer(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumIonServer::_GetSchemaKind() const
{
    return CesiumIonServer::schemaKind;
}

/* static */
const TfType &
CesiumIonServer::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumIonServer>();
    return tfType;
}

/* static */
bool 
CesiumIonServer::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumIonServer::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumIonServer::GetIonServerUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonServerUrl);
}

UsdAttribute
CesiumIonServer::CreateIonServerUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonServerUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumIonServer::GetIonServerApiUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonServerApiUrl);
}

UsdAttribute
CesiumIonServer::CreateIonServerApiUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonServerApiUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumIonServer::GetIonServerApplicationIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonServerApplicationId);
}

UsdAttribute
CesiumIonServer::CreateIonServerApplicationIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonServerApplicationId,
                       SdfValueTypeNames->Int64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumIonServer::GetProjectDefaultIonAccessTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumProjectDefaultIonAccessToken);
}

UsdAttribute
CesiumIonServer::CreateProjectDefaultIonAccessTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumProjectDefaultIonAccessToken,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumIonServer::GetProjectDefaultIonAccessTokenIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumProjectDefaultIonAccessTokenId);
}

UsdAttribute
CesiumIonServer::CreateProjectDefaultIonAccessTokenIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumProjectDefaultIonAccessTokenId,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

namespace {
static inline TfTokenVector
_ConcatenateAttributeNames(const TfTokenVector& left,const TfTokenVector& right)
{
    TfTokenVector result;
    result.reserve(left.size() + right.size());
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    return result;
}
}

/*static*/
const TfTokenVector&
CesiumIonServer::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumIonServerUrl,
        CesiumTokens->cesiumIonServerApiUrl,
        CesiumTokens->cesiumIonServerApplicationId,
        CesiumTokens->cesiumProjectDefaultIonAccessToken,
        CesiumTokens->cesiumProjectDefaultIonAccessTokenId,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            UsdTyped::GetSchemaAttributeNames(true),
            localNames);

    if (includeInherited)
        return allNames;
    else
        return localNames;
}

PXR_NAMESPACE_CLOSE_SCOPE

// ===================================================================== //
// Feel free to add custom code below this line. It will be preserved by
// the code generator.
//
// Just remember to wrap code in the appropriate delimiters:
// 'PXR_NAMESPACE_OPEN_SCOPE', 'PXR_NAMESPACE_CLOSE_SCOPE'.
// ===================================================================== //
// --(BEGIN CUSTOM CODE)--
