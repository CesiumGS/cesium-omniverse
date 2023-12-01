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

UsdRelationship
CesiumCartographicPolygon::GetBasisCurvesBindingRel() const
{
    return GetPrim().GetRelationship(CesiumTokens->cesiumBasisCurvesBinding);
}

UsdRelationship
CesiumCartographicPolygon::CreateBasisCurvesBindingRel() const
{
    return GetPrim().CreateRelationship(CesiumTokens->cesiumBasisCurvesBinding,
                       /* custom = */ false);
}

/*static*/
const TfTokenVector&
CesiumCartographicPolygon::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames;
    static TfTokenVector allNames =
        UsdGeomBasisCurves::GetSchemaAttributeNames(true);

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
