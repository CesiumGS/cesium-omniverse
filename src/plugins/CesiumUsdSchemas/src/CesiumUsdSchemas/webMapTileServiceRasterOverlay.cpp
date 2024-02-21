#include ".//webMapTileServiceRasterOverlay.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumWebMapTileServiceRasterOverlay,
        TfType::Bases< CesiumRasterOverlay > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumWebMapTileServiceRasterOverlayPrim")
    // to find TfType<CesiumWebMapTileServiceRasterOverlay>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumWebMapTileServiceRasterOverlay>("CesiumWebMapTileServiceRasterOverlayPrim");
}

/* virtual */
CesiumWebMapTileServiceRasterOverlay::~CesiumWebMapTileServiceRasterOverlay()
{
}

/* static */
CesiumWebMapTileServiceRasterOverlay
CesiumWebMapTileServiceRasterOverlay::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumWebMapTileServiceRasterOverlay();
    }
    return CesiumWebMapTileServiceRasterOverlay(stage->GetPrimAtPath(path));
}

/* static */
CesiumWebMapTileServiceRasterOverlay
CesiumWebMapTileServiceRasterOverlay::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumWebMapTileServiceRasterOverlayPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumWebMapTileServiceRasterOverlay();
    }
    return CesiumWebMapTileServiceRasterOverlay(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumWebMapTileServiceRasterOverlay::_GetSchemaKind() const
{
    return CesiumWebMapTileServiceRasterOverlay::schemaKind;
}

/* static */
const TfType &
CesiumWebMapTileServiceRasterOverlay::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumWebMapTileServiceRasterOverlay>();
    return tfType;
}

/* static */
bool 
CesiumWebMapTileServiceRasterOverlay::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumWebMapTileServiceRasterOverlay::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumUrl);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetLayerAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumLayer);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateLayerAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumLayer,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetStyleAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumStyle);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateStyleAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumStyle,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetFormatAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumFormat);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateFormatAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumFormat,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetTileMatrixSetIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTileMatrixSetId);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateTileMatrixSetIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTileMatrixSetId,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetSpecifyTileMatrixSetLabelsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSpecifyTileMatrixSetLabels);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateSpecifyTileMatrixSetLabelsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumSpecifyTileMatrixSetLabels,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetTileMatrixSetLabelPrefixAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTileMatrixSetLabelPrefix);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateTileMatrixSetLabelPrefixAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTileMatrixSetLabelPrefix,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetTileMatrixSetLabelsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTileMatrixSetLabels);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateTileMatrixSetLabelsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTileMatrixSetLabels,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetUseWebMercatorProjectionAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumUseWebMercatorProjection);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateUseWebMercatorProjectionAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumUseWebMercatorProjection,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetSpecifyTilingSchemeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSpecifyTilingScheme);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateSpecifyTilingSchemeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumSpecifyTilingScheme,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetRootTilesXAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumRootTilesX);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateRootTilesXAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumRootTilesX,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetRootTilesYAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumRootTilesY);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateRootTilesYAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumRootTilesY,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetWestAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumWest);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateWestAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumWest,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetEastAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEast);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateEastAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEast,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetSouthAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSouth);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateSouthAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumSouth,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetNorthAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumNorth);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateNorthAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumNorth,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetSpecifyZoomLevelsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSpecifyZoomLevels);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateSpecifyZoomLevelsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumSpecifyZoomLevels,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetMinimumZoomLevelAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMinimumZoomLevel);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateMinimumZoomLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMinimumZoomLevel,
                       SdfValueTypeNames->Int,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::GetMaximumZoomLevelAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumZoomLevel);
}

UsdAttribute
CesiumWebMapTileServiceRasterOverlay::CreateMaximumZoomLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
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
CesiumWebMapTileServiceRasterOverlay::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumUrl,
        CesiumTokens->cesiumLayer,
        CesiumTokens->cesiumStyle,
        CesiumTokens->cesiumFormat,
        CesiumTokens->cesiumTileMatrixSetId,
        CesiumTokens->cesiumSpecifyTileMatrixSetLabels,
        CesiumTokens->cesiumTileMatrixSetLabelPrefix,
        CesiumTokens->cesiumTileMatrixSetLabels,
        CesiumTokens->cesiumUseWebMercatorProjection,
        CesiumTokens->cesiumSpecifyTilingScheme,
        CesiumTokens->cesiumRootTilesX,
        CesiumTokens->cesiumRootTilesY,
        CesiumTokens->cesiumWest,
        CesiumTokens->cesiumEast,
        CesiumTokens->cesiumSouth,
        CesiumTokens->cesiumNorth,
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
