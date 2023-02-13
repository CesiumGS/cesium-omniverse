#include ".//rasterOverlay.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumRasterOverlay,
        TfType::Bases< UsdTyped > >();
    
}

/* virtual */
CesiumRasterOverlay::~CesiumRasterOverlay()
{
}

/* static */
CesiumRasterOverlay
CesiumRasterOverlay::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumRasterOverlay();
    }
    return CesiumRasterOverlay(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaType CesiumRasterOverlay::_GetSchemaType() const {
    return CesiumRasterOverlay::schemaType;
}

/* static */
const TfType &
CesiumRasterOverlay::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumRasterOverlay>();
    return tfType;
}

/* static */
bool 
CesiumRasterOverlay::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumRasterOverlay::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumRasterOverlay::GetRasterOverlayIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumRasterOverlayId);
}

UsdAttribute
CesiumRasterOverlay::CreateRasterOverlayIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumRasterOverlayId,
                       SdfValueTypeNames->Int64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumRasterOverlay::GetNameAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumName);
}

UsdAttribute
CesiumRasterOverlay::CreateNameAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumName,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumRasterOverlay::GetIonTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonToken);
}

UsdAttribute
CesiumRasterOverlay::CreateIonTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonToken,
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
CesiumRasterOverlay::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumRasterOverlayId,
        CesiumTokens->cesiumName,
        CesiumTokens->cesiumIonToken,
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
