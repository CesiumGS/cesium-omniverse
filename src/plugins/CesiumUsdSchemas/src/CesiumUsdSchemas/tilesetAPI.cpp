#include ".//tilesetAPI.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"
#include "pxr/usd/usd/tokens.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumTilesetAPI,
        TfType::Bases< UsdAPISchemaBase > >();
    
}

TF_DEFINE_PRIVATE_TOKENS(
    _schemaTokens,
    (CesiumTilesetSchemaAPI)
);

/* virtual */
CesiumTilesetAPI::~CesiumTilesetAPI()
{
}

/* static */
CesiumTilesetAPI
CesiumTilesetAPI::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumTilesetAPI();
    }
    return CesiumTilesetAPI(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaType CesiumTilesetAPI::_GetSchemaType() const {
    return CesiumTilesetAPI::schemaType;
}

/* static */
CesiumTilesetAPI
CesiumTilesetAPI::Apply(const UsdPrim &prim)
{
    if (prim.ApplyAPI<CesiumTilesetAPI>()) {
        return CesiumTilesetAPI(prim);
    }
    return CesiumTilesetAPI();
}

/* static */
const TfType &
CesiumTilesetAPI::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumTilesetAPI>();
    return tfType;
}

/* static */
bool 
CesiumTilesetAPI::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumTilesetAPI::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumTilesetAPI::GetUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumUrl);
}

UsdAttribute
CesiumTilesetAPI::CreateUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetIonAssetIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonAssetId);
}

UsdAttribute
CesiumTilesetAPI::CreateIonAssetIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonAssetId,
                       SdfValueTypeNames->Int64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetIonAccessTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonAccessToken);
}

UsdAttribute
CesiumTilesetAPI::CreateIonAccessTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumIonAccessToken,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetMaximumScreenSpaceErrorAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumScreenSpaceError);
}

UsdAttribute
CesiumTilesetAPI::CreateMaximumScreenSpaceErrorAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumScreenSpaceError,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetPreloadAncestorsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumPreloadAncestors);
}

UsdAttribute
CesiumTilesetAPI::CreatePreloadAncestorsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumPreloadAncestors,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetPreloadSiblingsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumPreloadSiblings);
}

UsdAttribute
CesiumTilesetAPI::CreatePreloadSiblingsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumPreloadSiblings,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetForbidHolesAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumForbidHoles);
}

UsdAttribute
CesiumTilesetAPI::CreateForbidHolesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumForbidHoles,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetMaximumSimultaneousTileLoadsAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumSimultaneousTileLoads);
}

UsdAttribute
CesiumTilesetAPI::CreateMaximumSimultaneousTileLoadsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumSimultaneousTileLoads,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetMaximumCachedBytesAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumMaximumCachedBytes);
}

UsdAttribute
CesiumTilesetAPI::CreateMaximumCachedBytesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumMaximumCachedBytes,
                       SdfValueTypeNames->UInt64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetLoadingDescendantLimitAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumLoadingDescendantLimit);
}

UsdAttribute
CesiumTilesetAPI::CreateLoadingDescendantLimitAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumLoadingDescendantLimit,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetEnableFrustumCullingAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEnableFrustumCulling);
}

UsdAttribute
CesiumTilesetAPI::CreateEnableFrustumCullingAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEnableFrustumCulling,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetEnableFogCullingAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEnableFogCulling);
}

UsdAttribute
CesiumTilesetAPI::CreateEnableFogCullingAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEnableFogCulling,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetEnforceCulledScreenSpaceErrorAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumEnforceCulledScreenSpaceError);
}

UsdAttribute
CesiumTilesetAPI::CreateEnforceCulledScreenSpaceErrorAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumEnforceCulledScreenSpaceError,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetCulledScreenSpaceErrorAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumCulledScreenSpaceError);
}

UsdAttribute
CesiumTilesetAPI::CreateCulledScreenSpaceErrorAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumCulledScreenSpaceError,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetSuspendUpdateAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumSuspendUpdate);
}

UsdAttribute
CesiumTilesetAPI::CreateSuspendUpdateAttr(VtValue const &defaultValue, bool writeSparsely) const
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
CesiumTilesetAPI::GetSchemaAttributeNames(bool includeInherited)
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
