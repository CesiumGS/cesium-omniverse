#include ".//ionImagery.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumIonImagery,
        TfType::Bases< CesiumImagery > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumIonImageryPrim")
    // to find TfType<CesiumIonImagery>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumIonImagery>("CesiumIonImageryPrim");
}

/* virtual */
CesiumIonImagery::~CesiumIonImagery()
{
}

/* static */
CesiumIonImagery
CesiumIonImagery::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumIonImagery();
    }
    return CesiumIonImagery(stage->GetPrimAtPath(path));
}

/* static */
CesiumIonImagery
CesiumIonImagery::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumIonImageryPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumIonImagery();
    }
    return CesiumIonImagery(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumIonImagery::_GetSchemaKind() const
{
    return CesiumIonImagery::schemaKind;
}

/* static */
const TfType &
CesiumIonImagery::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumIonImagery>();
    return tfType;
}

/* static */
bool 
CesiumIonImagery::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumIonImagery::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumIonImagery::GetIonAssetIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonAssetId);
}

UsdAttribute
CesiumIonImagery::CreateIonAssetIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonAssetId,
                       SdfValueTypeNames->Int64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumIonImagery::GetIonAccessTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonAccessToken);
}

UsdAttribute
CesiumIonImagery::CreateIonAccessTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonAccessToken,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdRelationship
CesiumIonImagery::GetIonServerBindingRel() const
{
    return GetPrim().GetRelationship(CesiumTokens->cesiumIonServerBinding);
}

UsdRelationship
CesiumIonImagery::CreateIonServerBindingRel() const
{
    return GetPrim().CreateRelationship(CesiumTokens->cesiumIonServerBinding,
                       /* custom = */ false);
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
CesiumIonImagery::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumIonAssetId,
        CesiumTokens->cesiumIonAccessToken,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            CesiumImagery::GetSchemaAttributeNames(true),
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
