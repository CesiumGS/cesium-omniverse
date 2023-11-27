#include ".//ionImagery.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<CesiumIonImagery,
        TfType::Bases< CesiumImagery > >();
    
    // Register the usd prim typename as an alias under UsdSchemaBase. This
    // enables one to call
    // TfType::Find<UsdSchemaBase>().FindDerivedByName("CesiumIonImageryPrim")
    // to find TfType<CesiumIonImagery>, which is how IsA queries are
    // answered.
    TfType::AddAlias<UsdSchemaBase, CesiumIonImagery>("CesiumIonImageryPrim");
}

/* virtual */
CesiumIonImagery::~CesiumIonImagery()
{
}

/* static */
CesiumIonImagery
CesiumIonImagery::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumIonImagery();
    }
    return CesiumIonImagery(stage->GetPrimAtPath(path));
}

/* static */
CesiumIonImagery
CesiumIonImagery::Define(
    const UsdStagePtr &stage, const SdfPath &path)
{
    static TfToken usdPrimTypeName("CesiumIonImageryPrim");
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return CesiumIonImagery();
    }
    return CesiumIonImagery(
        stage->DefinePrim(path, usdPrimTypeName));
}

/* virtual */
UsdSchemaKind CesiumIonImagery::_GetSchemaKind() const
{
    return CesiumIonImagery::schemaKind;
}

/* static */
const TfType &
CesiumIonImagery::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<CesiumIonImagery>();
    return tfType;
}

/* static */
bool 
CesiumIonImagery::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
CesiumIonImagery::_GetTfType() const
{
    return _GetStaticTfType();
}

/*static*/
const TfTokenVector&
CesiumIonImagery::GetSchemaAttributeNames(bool includeInherited)
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
