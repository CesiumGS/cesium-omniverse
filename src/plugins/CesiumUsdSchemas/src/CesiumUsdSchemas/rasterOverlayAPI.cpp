#include ".//rasterOverlayAPI.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"
#include "pxr/usd/usd/tokens.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumRasterOverlayAPI,
        TfType::Bases< UsdAPISchemaBase > >();
    
}

TF_DEFINE_PRIVATE_TOKENS(
    _schemaTokens,
    (RasterOverlayAPI)
);

/* virtual */
CesiumRasterOverlayAPI::~CesiumRasterOverlayAPI()
{
}

/* static */
CesiumRasterOverlayAPI
CesiumRasterOverlayAPI::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumRasterOverlayAPI();
    }
    return CesiumRasterOverlayAPI(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaType CesiumRasterOverlayAPI::_GetSchemaType() const {
    return CesiumRasterOverlayAPI::schemaType;
}

/* static */
CesiumRasterOverlayAPI
CesiumRasterOverlayAPI::Apply(const UsdPrim &prim)
{
    if (prim.ApplyAPI<CesiumRasterOverlayAPI>()) {
        return CesiumRasterOverlayAPI(prim);
    }
    return CesiumRasterOverlayAPI();
}

/* static */
const TfType &
CesiumRasterOverlayAPI::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumRasterOverlayAPI>();
    return tfType;
}

/* static */
bool 
CesiumRasterOverlayAPI::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumRasterOverlayAPI::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumRasterOverlayAPI::GetRasterOverlayIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumRasterOverlayId);
}

UsdAttribute
CesiumRasterOverlayAPI::CreateRasterOverlayIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumRasterOverlayId,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumRasterOverlayAPI::GetNameAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumName);
}

UsdAttribute
CesiumRasterOverlayAPI::CreateNameAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumName,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumRasterOverlayAPI::GetIonTokenIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonTokenId);
}

UsdAttribute
CesiumRasterOverlayAPI::CreateIonTokenIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonTokenId,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumRasterOverlayAPI::GetIonTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonToken);
}

UsdAttribute
CesiumRasterOverlayAPI::CreateIonTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
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
CesiumRasterOverlayAPI::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumRasterOverlayId,
        CesiumTokens->cesiumName,
        CesiumTokens->cesiumIonTokenId,
        CesiumTokens->cesiumIonToken,
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
