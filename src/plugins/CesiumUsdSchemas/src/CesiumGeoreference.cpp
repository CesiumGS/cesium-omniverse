#include "../include/cesium/omniverse/CesiumGeoreference.h"

#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumGeoreference,
        TfType::Bases< UsdTyped > >();

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


/* virtual */
UsdSchemaType CesiumGeoreference::_GetSchemaType() const {
    return CesiumGeoreference::schemaType;
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
CesiumGeoreference::GetLatitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->latitude);
}

UsdAttribute
CesiumGeoreference::CreateLatitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->latitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGeoreference::GetLongitudeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->longitude);
}

UsdAttribute
CesiumGeoreference::CreateLongitudeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->longitude,
                       SdfValueTypeNames->Double,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumGeoreference::GetHeightAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->height);
}

UsdAttribute
CesiumGeoreference::CreateHeightAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->height,
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
        CesiumTokens->latitude,
        CesiumTokens->longitude,
        CesiumTokens->height,
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
