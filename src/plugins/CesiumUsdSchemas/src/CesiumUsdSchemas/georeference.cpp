#include ".//georeference.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumGeoreference,
        TfType::Bases< UsdTyped > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumGeoreferencePrim")
    // to find TfType<CesiumGeoreference>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumGeoreference>("CesiumGeoreferencePrim");
}

/* virtual */
CesiumGeoreference::~CesiumGeoreference()
{
}

/* static */
CesiumGeoreference
CesiumGeoreference::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumGeoreference();
    }
    return CesiumGeoreference(stage->GetPrimAtPath(path));
}

/* static */
CesiumGeoreference
CesiumGeoreference::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumGeoreferencePrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumGeoreference();
    }
    return CesiumGeoreference(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumGeoreference::_GetSchemaKind() const
{
    return CesiumGeoreference::schemaKind;
}

/* static */
const TfType &
CesiumGeoreference::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumGeoreference>();
    return tfType;
}

/* static */
bool 
CesiumGeoreference::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumGeoreference::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumGeoreference::GetGeoreferenceOriginLongitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOriginLongitude);
}

UsdAttribute
CesiumGeoreference::CreateGeoreferenceOriginLongitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOriginLongitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGeoreference::GetGeoreferenceOriginLatitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOriginLatitude);
}

UsdAttribute
CesiumGeoreference::CreateGeoreferenceOriginLatitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOriginLatitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGeoreference::GetGeoreferenceOriginHeightAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumGeoreferenceOriginHeight);
}

UsdAttribute
CesiumGeoreference::CreateGeoreferenceOriginHeightAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumGeoreferenceOriginHeight,
                       SdfValueTypeNames->Double,
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
CesiumGeoreference::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumGeoreferenceOriginLongitude,
        CesiumTokens->cesiumGeoreferenceOriginLatitude,
        CesiumTokens->cesiumGeoreferenceOriginHeight,
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
