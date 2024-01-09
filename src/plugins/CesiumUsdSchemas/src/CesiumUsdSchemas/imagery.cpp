#include ".//imagery.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumImagery,
        TfType::Bases< UsdTyped > >();
    
}

/* virtual */
CesiumImagery::~CesiumImagery()
{
}

/* static */
CesiumImagery
CesiumImagery::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumImagery();
    }
    return CesiumImagery(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaKind CesiumImagery::_GetSchemaKind() const
{
    return CesiumImagery::schemaKind;
}

/* static */
const TfType &
CesiumImagery::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumImagery>();
    return tfType;
}

/* static */
bool 
CesiumImagery::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumImagery::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumImagery::GetShowCreditsOnScreenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumShowCreditsOnScreen);
}

UsdAttribute
CesiumImagery::CreateShowCreditsOnScreenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumShowCreditsOnScreen,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumImagery::GetAlphaAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumAlpha);
}

UsdAttribute
CesiumImagery::CreateAlphaAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumAlpha,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumImagery::GetOverlayRenderPipeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumOverlayRenderPipe);
}

UsdAttribute
CesiumImagery::CreateOverlayRenderPipeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumOverlayRenderPipe,
                       SdfValueTypeNames->Token,
                       /* custom = */ false,
                       SdfVariabilityUniform,
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
CesiumImagery::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumShowCreditsOnScreen,
        CesiumTokens->cesiumAlpha,
        CesiumTokens->cesiumOverlayRenderPipe,
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
