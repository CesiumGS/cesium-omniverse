#include ".//polygonRasterOverlay.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumPolygonRasterOverlay,
        TfType::Bases< CesiumRasterOverlay > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumPolygonRasterOverlayPrim")
    // to find TfType<CesiumPolygonRasterOverlay>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumPolygonRasterOverlay>("CesiumPolygonRasterOverlayPrim");
}

/* virtual */
CesiumPolygonRasterOverlay::~CesiumPolygonRasterOverlay()
{
}

/* static */
CesiumPolygonRasterOverlay
CesiumPolygonRasterOverlay::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumPolygonRasterOverlay();
    }
    return CesiumPolygonRasterOverlay(stage->GetPrimAtPath(path));
}

/* static */
CesiumPolygonRasterOverlay
CesiumPolygonRasterOverlay::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumPolygonRasterOverlayPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumPolygonRasterOverlay();
    }
    return CesiumPolygonRasterOverlay(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumPolygonRasterOverlay::_GetSchemaKind() const
{
    return CesiumPolygonRasterOverlay::schemaKind;
}

/* static */
const TfType &
CesiumPolygonRasterOverlay::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumPolygonRasterOverlay>();
    return tfType;
}

/* static */
bool 
CesiumPolygonRasterOverlay::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumPolygonRasterOverlay::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdRelationship
CesiumPolygonRasterOverlay::GetCartographicPolygonBindingRel() const
{
    return GetPrim().GetRelationship(CesiumTokens->cesiumCartographicPolygonBinding);
}

UsdRelationship
CesiumPolygonRasterOverlay::CreateCartographicPolygonBindingRel() const
{
    return GetPrim().CreateRelationship(CesiumTokens->cesiumCartographicPolygonBinding,
                       /* custom = */ false);
}

/*static*/
const TfTokenVector&
CesiumPolygonRasterOverlay::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames;
    static TfTokenVector allNames =
        CesiumRasterOverlay::GetSchemaAttributeNames(true);

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
