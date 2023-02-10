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
CesiumData::GetCesiumDefaultProjectTokenIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumDefaultProjectTokenId);
}

UsdAttribute
CesiumData::CreateCesiumDefaultProjectTokenIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumDefaultProjectTokenId,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumData::GetCesiumDefaultProjectTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumDefaultProjectToken);
}

UsdAttribute
CesiumData::CreateCesiumDefaultProjectTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumDefaultProjectToken,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumData::GetCesiumGeoreferenceOriginAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOrigin);
}

UsdAttribute
CesiumData::CreateCesiumGeoreferenceOriginAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOrigin,
                       SdfValueTypeNames->Double3,
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
        CesiumTokens->cesiumDefaultProjectTokenId,
        CesiumTokens->cesiumDefaultProjectToken,
        CesiumTokens->cesiumGeoreferenceOrigin,
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
