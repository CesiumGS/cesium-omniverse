#include ".//ellipsoid.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumEllipsoid,
        TfType::Bases< UsdTyped > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumEllipsoidPrim")
    // to find TfType<CesiumEllipsoid>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumEllipsoid>("CesiumEllipsoidPrim");
}

/* virtual */
CesiumEllipsoid::~CesiumEllipsoid()
{
}

/* static */
CesiumEllipsoid
CesiumEllipsoid::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumEllipsoid();
    }
    return CesiumEllipsoid(stage->GetPrimAtPath(path));
}

/* static */
CesiumEllipsoid
CesiumEllipsoid::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumEllipsoidPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumEllipsoid();
    }
    return CesiumEllipsoid(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumEllipsoid::_GetSchemaKind() const
{
    return CesiumEllipsoid::schemaKind;
}

/* static */
const TfType &
CesiumEllipsoid::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumEllipsoid>();
    return tfType;
}

/* static */
bool 
CesiumEllipsoid::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumEllipsoid::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumEllipsoid::GetRadiiAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->cesiumRadii);
}

UsdAttribute
CesiumEllipsoid::CreateRadiiAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->cesiumRadii,
                       SdfValueTypeNames->Double3,
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
CesiumEllipsoid::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->cesiumRadii,
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
