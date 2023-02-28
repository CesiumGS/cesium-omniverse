#include ".//tileset.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"
#include "pxr/usd/usd/tokens.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumTileset,
        TfType::Bases< UsdAPISchemaBase > >();
    
}

TF_DEFINE_PRIVATE_TOKENS(
    _schemaTokens,
    (CesiumTilesetAPI)
);

/* virtual */
CesiumTileset::~CesiumTileset()
{
}

/* static */
CesiumTileset
CesiumTileset::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumTileset();
    }
    return CesiumTileset(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaType CesiumTileset::_GetSchemaType() const {
    return CesiumTileset::schemaType;
}

/* static */
CesiumTileset
CesiumTileset::Apply(const UsdPrim &prim)
{
    if (prim.ApplyAPI<CesiumTileset>()) {
        return CesiumTileset(prim);
    }
    return CesiumTileset();
}

/* static */
const TfType &
CesiumTileset::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumTileset>();
    return tfType;
}

/* static */
bool 
CesiumTileset::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumTileset::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumTileset::GetUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumUrl);
}

UsdAttribute
CesiumTileset::CreateUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetIonAssetIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonAssetId);
}

UsdAttribute
CesiumTileset::CreateIonAssetIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonAssetId,
                       SdfValueTypeNames->Int64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetIonAccessTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonAccessToken);
}

UsdAttribute
CesiumTileset::CreateIonAccessTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonAccessToken,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetMaximumScreenSpaceErrorAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumScreenSpaceError);
}

UsdAttribute
CesiumTileset::CreateMaximumScreenSpaceErrorAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumScreenSpaceError,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetPreloadAncestorsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumPreloadAncestors);
}

UsdAttribute
CesiumTileset::CreatePreloadAncestorsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumPreloadAncestors,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetPreloadSiblingsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumPreloadSiblings);
}

UsdAttribute
CesiumTileset::CreatePreloadSiblingsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumPreloadSiblings,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetForbidHolesAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumForbidHoles);
}

UsdAttribute
CesiumTileset::CreateForbidHolesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumForbidHoles,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetMaximumSimultaneousTileLoadsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumSimultaneousTileLoads);
}

UsdAttribute
CesiumTileset::CreateMaximumSimultaneousTileLoadsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumSimultaneousTileLoads,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetMaximumCachedBytesAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumCachedBytes);
}

UsdAttribute
CesiumTileset::CreateMaximumCachedBytesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumCachedBytes,
                       SdfValueTypeNames->UInt64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetLoadingDescendantLimitAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumLoadingDescendantLimit);
}

UsdAttribute
CesiumTileset::CreateLoadingDescendantLimitAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumLoadingDescendantLimit,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetEnableFrustumCullingAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEnableFrustumCulling);
}

UsdAttribute
CesiumTileset::CreateEnableFrustumCullingAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEnableFrustumCulling,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetEnableFogCullingAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEnableFogCulling);
}

UsdAttribute
CesiumTileset::CreateEnableFogCullingAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEnableFogCulling,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetEnforceCulledScreenSpaceErrorAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEnforceCulledScreenSpaceError);
}

UsdAttribute
CesiumTileset::CreateEnforceCulledScreenSpaceErrorAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEnforceCulledScreenSpaceError,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetCulledScreenSpaceErrorAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumCulledScreenSpaceError);
}

UsdAttribute
CesiumTileset::CreateCulledScreenSpaceErrorAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumCulledScreenSpaceError,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTileset::GetSuspendUpdateAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSuspendUpdate);
}

UsdAttribute
CesiumTileset::CreateSuspendUpdateAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumSuspendUpdate,
                       SdfValueTypeNames->Bool,
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
CesiumTileset::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumUrl,
        CesiumTokens->cesiumIonAssetId,
        CesiumTokens->cesiumIonAccessToken,
        CesiumTokens->cesiumMaximumScreenSpaceError,
        CesiumTokens->cesiumPreloadAncestors,
        CesiumTokens->cesiumPreloadSiblings,
        CesiumTokens->cesiumForbidHoles,
        CesiumTokens->cesiumMaximumSimultaneousTileLoads,
        CesiumTokens->cesiumMaximumCachedBytes,
        CesiumTokens->cesiumLoadingDescendantLimit,
        CesiumTokens->cesiumEnableFrustumCulling,
        CesiumTokens->cesiumEnableFogCulling,
        CesiumTokens->cesiumEnforceCulledScreenSpaceError,
        CesiumTokens->cesiumCulledScreenSpaceError,
        CesiumTokens->cesiumSuspendUpdate,
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
