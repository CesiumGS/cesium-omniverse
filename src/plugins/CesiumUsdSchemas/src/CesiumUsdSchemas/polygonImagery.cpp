#include ".//polygonImagery.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumPolygonImagery,
        TfType::Bases< CesiumImagery > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumPolygonImageryPrim")
    // to find TfType<CesiumPolygonImagery>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumPolygonImagery>("CesiumPolygonImageryPrim");
}

/* virtual */
CesiumPolygonImagery::~CesiumPolygonImagery()
{
}

/* static */
CesiumPolygonImagery
CesiumPolygonImagery::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumPolygonImagery();
    }
    return CesiumPolygonImagery(stage->GetPrimAtPath(path));
}

/* static */
CesiumPolygonImagery
CesiumPolygonImagery::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumPolygonImageryPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumPolygonImagery();
    }
    return CesiumPolygonImagery(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumPolygonImagery::_GetSchemaKind() const
{
    return CesiumPolygonImagery::schemaKind;
}

/* static */
const TfType &
CesiumPolygonImagery::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumPolygonImagery>();
    return tfType;
}

/* static */
bool 
CesiumPolygonImagery::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumPolygonImagery::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdRelationship
CesiumPolygonImagery::GetCartographicPolygonBindingRel() const
{
    return GetPrim().GetRelationship(CesiumTokens->cesiumCartographicPolygonBinding);
}

UsdRelationship
CesiumPolygonImagery::CreateCartographicPolygonBindingRel() const
{
    return GetPrim().CreateRelationship(CesiumTokens->cesiumCartographicPolygonBinding,
                       /* custom = */ false);
}

UsdRelationship
CesiumPolygonImagery::GetGlobeAnchorBindingRel() const
{
    return GetPrim().GetRelationship(CesiumTokens->cesiumGlobeAnchorBinding);
}

UsdRelationship
CesiumPolygonImagery::CreateGlobeAnchorBindingRel() const
{
    return GetPrim().CreateRelationship(CesiumTokens->cesiumGlobeAnchorBinding,
                       /* custom = */ false);
}

/*static*/
const TfTokenVector&
CesiumPolygonImagery::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames;
    static TfTokenVector allNames =
        CesiumImagery::GetSchemaAttributeNames(true);

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
