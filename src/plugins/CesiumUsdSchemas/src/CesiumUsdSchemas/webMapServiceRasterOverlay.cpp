#include ".//webMapServiceRasterOverlay.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumWebMapServiceRasterOverlay,
        TfType::Bases< CesiumRasterOverlay > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumWebMapServiceRasterOverlayPrim")
    // to find TfType<CesiumWebMapServiceRasterOverlay>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumWebMapServiceRasterOverlay>("CesiumWebMapServiceRasterOverlayPrim");
}

/* virtual */
CesiumWebMapServiceRasterOverlay::~CesiumWebMapServiceRasterOverlay()
{
}

/* static */
CesiumWebMapServiceRasterOverlay
CesiumWebMapServiceRasterOverlay::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumWebMapServiceRasterOverlay();
    }
    return CesiumWebMapServiceRasterOverlay(stage->GetPrimAtPath(path));
}

/* static */
CesiumWebMapServiceRasterOverlay
CesiumWebMapServiceRasterOverlay::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumWebMapServiceRasterOverlayPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumWebMapServiceRasterOverlay();
    }
    return CesiumWebMapServiceRasterOverlay(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumWebMapServiceRasterOverlay::_GetSchemaKind() const
{
    return CesiumWebMapServiceRasterOverlay::schemaKind;
}

/* static */
const TfType &
CesiumWebMapServiceRasterOverlay::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumWebMapServiceRasterOverlay>();
    return tfType;
}

/* static */
bool 
CesiumWebMapServiceRasterOverlay::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumWebMapServiceRasterOverlay::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::GetBaseUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumBaseUrl);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::CreateBaseUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumBaseUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::GetLayersAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumLayers);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::CreateLayersAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumLayers,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::GetTileWidthAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTileWidth);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::CreateTileWidthAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTileWidth,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::GetTileHeightAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTileHeight);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::CreateTileHeightAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTileHeight,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::GetMinimumLevelAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMinimumLevel);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::CreateMinimumLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMinimumLevel,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::GetMaximumLevelAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumLevel);
}

UsdAttribute
CesiumWebMapServiceRasterOverlay::CreateMaximumLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumLevel,
                       SdfValueTypeNames->Int,
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
CesiumWebMapServiceRasterOverlay::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumBaseUrl,
        CesiumTokens->cesiumLayers,
        CesiumTokens->cesiumTileWidth,
        CesiumTokens->cesiumTileHeight,
        CesiumTokens->cesiumMinimumLevel,
        CesiumTokens->cesiumMaximumLevel,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            CesiumRasterOverlay::GetSchemaAttributeNames(true),
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
