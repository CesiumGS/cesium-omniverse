#include ".//globalAnchorAPI.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"
#include "pxr/usd/usd/tokens.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumGlobalAnchorAPI,
        TfType::Bases< UsdAPISchemaBase > >();
    
}

TF_DEFINE_PRIVATE_TOKENS(
    _schemaTokens,
    (CesiumGlobalAnchorAPI)
);

/* virtual */
CesiumGlobalAnchorAPI::~CesiumGlobalAnchorAPI()
{
}

/* static */
CesiumGlobalAnchorAPI
CesiumGlobalAnchorAPI::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumGlobalAnchorAPI();
    }
    return CesiumGlobalAnchorAPI(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaType CesiumGlobalAnchorAPI::_GetSchemaType() const {
    return CesiumGlobalAnchorAPI::schemaType;
}

/* static */
CesiumGlobalAnchorAPI
CesiumGlobalAnchorAPI::Apply(const UsdPrim &prim)
{
    if (prim.ApplyAPI<CesiumGlobalAnchorAPI>()) {
        return CesiumGlobalAnchorAPI(prim);
    }
    return CesiumGlobalAnchorAPI();
}

/* static */
const TfType &
CesiumGlobalAnchorAPI::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumGlobalAnchorAPI>();
    return tfType;
}

/* static */
bool 
CesiumGlobalAnchorAPI::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumGlobalAnchorAPI::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumGlobalAnchorAPI::GetAdjustOrientationForGlobeWhenMovingAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateAdjustOrientationForGlobeWhenMovingAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetDetectTransformChangesAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorDetectTransformChanges);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateDetectTransformChangesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorDetectTransformChanges,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetLongitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorLongitude);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateLongitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorLongitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetLatitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorLatitude);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateLatitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorLatitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetHeightAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorHeight);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateHeightAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorHeight,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetPositionAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorPosition);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreatePositionAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorPosition,
                       SdfValueTypeNames->Double3,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetRotationAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorRotation);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateRotationAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorRotation,
                       SdfValueTypeNames->Double3,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobalAnchorAPI::GetScaleAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorScale);
}

UsdAttribute
CesiumGlobalAnchorAPI::CreateScaleAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorScale,
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
CesiumGlobalAnchorAPI::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving,
        CesiumTokens->cesiumAnchorDetectTransformChanges,
        CesiumTokens->cesiumAnchorLongitude,
        CesiumTokens->cesiumAnchorLatitude,
        CesiumTokens->cesiumAnchorHeight,
        CesiumTokens->cesiumAnchorPosition,
        CesiumTokens->cesiumAnchorRotation,
        CesiumTokens->cesiumAnchorScale,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            UsdAPISchemaBase::GetSchemaAttributeNames(true),
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
