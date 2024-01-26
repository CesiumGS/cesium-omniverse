#include ".//cartographicPolygon.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumCartographicPolygon,
        TfType::Bases< UsdGeomBasisCurves > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumCartographicPolygonPrim")
    // to find TfType<CesiumCartographicPolygon>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumCartographicPolygon>("CesiumCartographicPolygonPrim");
}

/* virtual */
CesiumCartographicPolygon::~CesiumCartographicPolygon()
{
}

/* static */
CesiumCartographicPolygon
CesiumCartographicPolygon::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumCartographicPolygon();
    }
    return CesiumCartographicPolygon(stage->GetPrimAtPath(path));
}

/* static */
CesiumCartographicPolygon
CesiumCartographicPolygon::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumCartographicPolygonPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumCartographicPolygon();
    }
    return CesiumCartographicPolygon(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumCartographicPolygon::_GetSchemaKind() const
{
    return CesiumCartographicPolygon::schemaKind;
}

/* static */
const TfType &
CesiumCartographicPolygon::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumCartographicPolygon>();
    return tfType;
}

/* static */
bool 
CesiumCartographicPolygon::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumCartographicPolygon::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
CesiumCartographicPolygon::GetTypeAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->type);
}

UsdAttribute
CesiumCartographicPolygon::CreateTypeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->type,
                       SdfValueTypeNames->Token,
                       /* custom = */ false,
                       SdfVariabilityUniform,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
CesiumCartographicPolygon::GetWrapAttr() const
{
    return GetPrim().GetAttribute(CesiumTokens->wrap);
}

UsdAttribute
CesiumCartographicPolygon::CreateWrapAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(CesiumTokens->wrap,
                       SdfValueTypeNames->Token,
                       /* custom = */ false,
                       SdfVariabilityUniform,
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
CesiumCartographicPolygon::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        CesiumTokens->type,
        CesiumTokens->wrap,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            UsdGeomBasisCurves::GetSchemaAttributeNames(true),
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
