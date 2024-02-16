#include ".//tileMapServiceRasterOverlay.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumTileMapServiceRasterOverlay,
        TfType::Bases< CesiumRasterOverlay > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumTileMapServiceRasterOverlayPrim")
    // to find TfType<CesiumTileMapServiceRasterOverlay>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumTileMapServiceRasterOverlay>("CesiumTileMapServiceRasterOverlayPrim");
}

/* virtual */
CesiumTileMapServiceRasterOverlay::~CesiumTileMapServiceRasterOverlay()
{
}

/* static */
CesiumTileMapServiceRasterOverlay
CesiumTileMapServiceRasterOverlay::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumTileMapServiceRasterOverlay();
    }
    return CesiumTileMapServiceRasterOverlay(stage->GetPrimAtPath(path));
}

/* static */
CesiumTileMapServiceRasterOverlay
CesiumTileMapServiceRasterOverlay::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumTileMapServiceRasterOverlayPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumTileMapServiceRasterOverlay();
    }
    return CesiumTileMapServiceRasterOverlay(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumTileMapServiceRasterOverlay::_GetSchemaKind() const
{
    return CesiumTileMapServiceRasterOverlay::schemaKind;
}

/* static */
const TfType &
CesiumTileMapServiceRasterOverlay::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumTileMapServiceRasterOverlay>();
    return tfType;
}

/* static */
bool 
CesiumTileMapServiceRasterOverlay::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumTileMapServiceRasterOverlay::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::GetUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumUrl);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::CreateUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::GetSpecifyZoomLevelsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSpecifyZoomLevels);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::CreateSpecifyZoomLevelsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumSpecifyZoomLevels,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::GetMinimumZoomLevelAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMinimumZoomLevel);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::CreateMinimumZoomLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMinimumZoomLevel,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::GetMaximumZoomLevelAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumZoomLevel);
}

UsdAttribute
CesiumTileMapServiceRasterOverlay::CreateMaximumZoomLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumZoomLevel,
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
CesiumTileMapServiceRasterOverlay::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumUrl,
        CesiumTokens->cesiumSpecifyZoomLevels,
        CesiumTokens->cesiumMinimumZoomLevel,
        CesiumTokens->cesiumMaximumZoomLevel,
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
