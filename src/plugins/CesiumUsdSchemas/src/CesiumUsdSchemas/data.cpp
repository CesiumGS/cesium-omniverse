#include ".//data.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumData,
        TfType::Bases< UsdTyped > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumDataPrim")
    // to find TfType<CesiumData>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumData>("CesiumDataPrim");
}

/* virtual */
CesiumData::~CesiumData()
{
}

/* static */
CesiumData
CesiumData::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumData();
    }
    return CesiumData(stage->GetPrimAtPath(path));
}

/* static */
CesiumData
CesiumData::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumDataPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumData();
    }
    return CesiumData(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaType CesiumData::_GetSchemaType() const {
    return CesiumData::schemaType;
}

/* static */
const TfType &
CesiumData::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumData>();
    return tfType;
}

/* static */
bool 
CesiumData::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumData::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumData::GetDefaultProjectIonAccessTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumDefaultProjectIonAccessToken);
}

UsdAttribute
CesiumData::CreateDefaultProjectIonAccessTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumDefaultProjectIonAccessToken,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumData::GetDefaultProjectIonAccessTokenIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumDefaultProjectIonAccessTokenId);
}

UsdAttribute
CesiumData::CreateDefaultProjectIonAccessTokenIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumDefaultProjectIonAccessTokenId,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumData::GetGeoreferenceOriginLongitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOriginLongitude);
}

UsdAttribute
CesiumData::CreateGeoreferenceOriginLongitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOriginLongitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumData::GetGeoreferenceOriginLatitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOriginLatitude);
}

UsdAttribute
CesiumData::CreateGeoreferenceOriginLatitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOriginLatitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumData::GetGeoreferenceOriginHeightAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOriginHeight);
}

UsdAttribute
CesiumData::CreateGeoreferenceOriginHeightAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOriginHeight,
                       SdfValueTypeNames->Double,
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
CesiumData::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumDefaultProjectIonAccessToken,
        CesiumTokens->cesiumDefaultProjectIonAccessTokenId,
        CesiumTokens->cesiumGeoreferenceOriginLongitude,
        CesiumTokens->cesiumGeoreferenceOriginLatitude,
        CesiumTokens->cesiumGeoreferenceOriginHeight,
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
