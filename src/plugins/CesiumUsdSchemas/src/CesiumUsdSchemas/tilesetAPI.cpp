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
    (TilesetAPI)
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
CesiumTilesetAPI::GetTilesetIdAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTilesetId);
}

UsdAttribute
CesiumTilesetAPI::CreateTilesetIdAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTilesetId,
                       SdfValueTypeNames->Int64,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetTilesetUrlAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumTilesetUrl);
}

UsdAttribute
CesiumTilesetAPI::CreateTilesetUrlAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumTilesetUrl,
                       SdfValueTypeNames->String,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumTilesetAPI::GetIonTokenAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumIonToken);
}

UsdAttribute
CesiumTilesetAPI::CreateIonTokenAttr(VtValue const &defaultValue, bool writeSparsely) const
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
CesiumTilesetAPI::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumTilesetId,
        CesiumTokens->cesiumTilesetUrl,
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
