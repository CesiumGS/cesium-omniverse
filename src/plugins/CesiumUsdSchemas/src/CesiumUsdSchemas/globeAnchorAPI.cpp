#include ".//globeAnchorAPI.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"
#include "pxr/usd/usd/tokens.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumGlobeAnchorAPI,
        TfType::Bases< UsdAPISchemaBase > >();
    
}

TF_DEFINE_PRIVATE_TOKENS(
    _schemaTokens,
    (CesiumGlobeAnchorSchemaAPI)
);

/* virtual */
CesiumGlobeAnchorAPI::~CesiumGlobeAnchorAPI()
{
}

/* static */
CesiumGlobeAnchorAPI
CesiumGlobeAnchorAPI::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumGlobeAnchorAPI();
    }
    return CesiumGlobeAnchorAPI(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaKind CesiumGlobeAnchorAPI::_GetSchemaKind() const
{
    return CesiumGlobeAnchorAPI::schemaKind;
}

/* static */
bool
CesiumGlobeAnchorAPI::CanApply(
    const UsdPrim &prim, std::string *whyNot)
{
    return prim.CanApplyAPI<CesiumGlobeAnchorAPI>(whyNot);
}

/* static */
CesiumGlobeAnchorAPI
CesiumGlobeAnchorAPI::Apply(const UsdPrim &prim)
{
    if (prim.ApplyAPI<CesiumGlobeAnchorAPI>()) {
        return CesiumGlobeAnchorAPI(prim);
    }
    return CesiumGlobeAnchorAPI();
}

/* static */
const TfType &
CesiumGlobeAnchorAPI::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumGlobeAnchorAPI>();
    return tfType;
}

/* static */
bool 
CesiumGlobeAnchorAPI::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumGlobeAnchorAPI::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumGlobeAnchorAPI::GetAdjustOrientationForGlobeWhenMovingAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving);
}

UsdAttribute
CesiumGlobeAnchorAPI::CreateAdjustOrientationForGlobeWhenMovingAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobeAnchorAPI::GetDetectTransformChangesAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorDetectTransformChanges);
}

UsdAttribute
CesiumGlobeAnchorAPI::CreateDetectTransformChangesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorDetectTransformChanges,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobeAnchorAPI::GetGeographicCoordinateAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorGeographicCoordinates);
}

UsdAttribute
CesiumGlobeAnchorAPI::CreateGeographicCoordinateAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorGeographicCoordinates,
                       SdfValueTypeNames->Double3,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobeAnchorAPI::GetPositionAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorPosition);
}

UsdAttribute
CesiumGlobeAnchorAPI::CreatePositionAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorPosition,
                       SdfValueTypeNames->Double3,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobeAnchorAPI::GetRotationAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorRotation);
}

UsdAttribute
CesiumGlobeAnchorAPI::CreateRotationAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorRotation,
                       SdfValueTypeNames->Double3,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGlobeAnchorAPI::GetScaleAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAnchorScale);
}

UsdAttribute
CesiumGlobeAnchorAPI::CreateScaleAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAnchorScale,
                       SdfValueTypeNames->Double3,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdRelationship
CesiumGlobeAnchorAPI::GetGeoreferenceBindingRel() const
{
    return GetPrim().GetRelationship(CesiumTokens->cesiumAnchorGeoreferenceBinding);
}

UsdRelationship
CesiumGlobeAnchorAPI::CreateGeoreferenceBindingRel() const
{
    return GetPrim().CreateRelationship(CesiumTokens->cesiumAnchorGeoreferenceBinding,
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
CesiumGlobeAnchorAPI::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumAnchorAdjustOrientationForGlobeWhenMoving,
        CesiumTokens->cesiumAnchorDetectTransformChanges,
        CesiumTokens->cesiumAnchorGeographicCoordinates,
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
