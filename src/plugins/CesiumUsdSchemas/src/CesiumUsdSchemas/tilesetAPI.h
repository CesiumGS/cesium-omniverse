#ifndef CESIUM_GENERATED_TILESETAPI_H
#define CESIUM_GENERATED_TILESETAPI_H

/// \file cesium/tilesetAPI.h

#include "pxr/pxr.h"
#include ".//api.h"
#include "pxr/usd/usd/apiSchemaBase.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include ".//tokens.h"

#include "pxr/base/vt/value.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/matrix4d.h"

#include "pxr/base/tf/token.h"
#include "pxr/base/tf/type.h"

PXR_NAMESPACE_OPEN_SCOPE

class SdfAssetPath;

// -------------------------------------------------------------------------- //
// CESIUMTILESETSCHEMAAPI                                                     //
// -------------------------------------------------------------------------- //

/// \class CesiumTilesetAPI
///
/// Adds Cesium specific data to a prim for representing a tileset.
///
/// For any described attribute \em Fallback \em Value or \em Allowed \em Values below
/// that are text/tokens, the actual token is published and defined in \ref CesiumTokens.
/// So to set an attribute to the value "rightHanded", use CesiumTokens->rightHanded
/// as the value.
///
class CESIUM_API CesiumTilesetAPI : public UsdAPISchemaBase
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaType
    static const UsdSchemaType schemaType = UsdSchemaType::SingleApplyAPI;

    /// Construct a CesiumTilesetAPI on UsdPrim \p prim .
    /// Equivalent to CesiumTilesetAPI::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumTilesetAPI(const UsdPrim& prim=UsdPrim())
        : UsdAPISchemaBase(prim)
    {
    }

    /// Construct a CesiumTilesetAPI on the prim held by \p schemaObj .
    /// Should be preferred over CesiumTilesetAPI(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumTilesetAPI(const UsdSchemaBase& schemaObj)
        : UsdAPISchemaBase(schemaObj)
    {
    }

    /// Destructor.
    virtual ~CesiumTilesetAPI();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumTilesetAPI holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumTilesetAPI(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    static CesiumTilesetAPI
    Get(const UsdStagePtr &stage, const SdfPath &path);


    /// Applies this <b>single-apply</b> API schema to the given \p prim.
    /// This information is stored by adding "CesiumTilesetSchemaAPI" to the 
    /// token-valued, listOp metadata \em apiSchemas on the prim.
    /// 
    /// \return A valid CesiumTilesetAPI object is returned upon success. 
    /// An invalid (or empty) CesiumTilesetAPI object is returned upon 
    /// failure. See \ref UsdPrim::ApplyAPI() for conditions 
    /// resulting in failure. 
    /// 
    /// \sa UsdPrim::GetAppliedSchemas()
    /// \sa UsdPrim::HasAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///
    static CesiumTilesetAPI 
    Apply(const UsdPrim &prim);

protected:
    /// Returns the type of schema this class belongs to.
    ///
    /// \sa UsdSchemaType
    UsdSchemaType _GetSchemaType() const override;

private:
    // needs to invoke _GetStaticTfType.
    friend class UsdSchemaRegistry;
    static const TfType &_GetStaticTfType();

    static bool _IsTypedSchema();

    // override SchemaBase virtuals.
    const TfType &_GetTfType() const override;

public:
    // --------------------------------------------------------------------- //
    // SOURCETYPE 
    // --------------------------------------------------------------------- //
    /// Selects whether to use the Cesium ion Asset ID or the provided URL for this tileset.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform token cesium:sourceType = "ion"` |
    /// | C++ Type | TfToken |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Token |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    /// | \ref CesiumTokens "Allowed Values" | ion, url |
    UsdAttribute GetSourceTypeAttr() const;

    /// See GetSourceTypeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateSourceTypeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // URL 
    // --------------------------------------------------------------------- //
    /// The URL of this tileset's tileset.json file. Usually blank if this is an ion asset.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:url = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetUrlAttr() const;

    /// See GetUrlAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateUrlAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // IONASSETID 
    // --------------------------------------------------------------------- //
    /// The ID of the Cesium ion asset to use. Usually blank if using URL.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `int64 cesium:ionAssetId = 0` |
    /// | C++ Type | int64_t |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int64 |
    UsdAttribute GetIonAssetIdAttr() const;

    /// See GetIonAssetIdAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateIonAssetIdAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // IONACCESSTOKEN 
    // --------------------------------------------------------------------- //
    /// The access token to use to access the Cesium ion resource. Overrides the default token. Usually blank if using URL.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:ionAccessToken = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetIonAccessTokenAttr() const;

    /// See GetIonAccessTokenAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateIonAccessTokenAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMSCREENSPACEERROR 
    // --------------------------------------------------------------------- //
    /// The maximum number of pixels of error when rendering this tileset. This is used to select an appropriate level-of-detail: A low value will cause many tiles with a high level of detail to be loaded, causing a finer visual representation of the tiles, but with a higher performance cost for loading and rendering. A higher value will cause a coarser visual representation, with lower performance requirements. When a tileset uses the older layer.json / quantized-mesh format rather than 3D Tiles, this value is effectively divided by 8.0. So the default value of 16.0 corresponds to the standard value for quantized-mesh terrain of 2.0.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `float cesium:maximumScreenSpaceError = 16` |
    /// | C++ Type | float |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float |
    UsdAttribute GetMaximumScreenSpaceErrorAttr() const;

    /// See GetMaximumScreenSpaceErrorAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateMaximumScreenSpaceErrorAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRELOADANCESTORS 
    // --------------------------------------------------------------------- //
    /// Whether to preload ancestor tiles. Setting this to true optimizes the zoom-out experience and provides more detail in newly-exposed areas when panning. The down side is that it requires loading more tiles.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:preloadAncestors = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetPreloadAncestorsAttr() const;

    /// See GetPreloadAncestorsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreatePreloadAncestorsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRELOADSIBLINGS 
    // --------------------------------------------------------------------- //
    /// Whether to preload sibling tiles. Setting this to true causes tiles with the same parent as a rendered tile to be loaded, even if they are culled. Setting this to true may provide a better panning experience at the cost of loading more tiles.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:preloadSiblings = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetPreloadSiblingsAttr() const;

    /// See GetPreloadSiblingsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreatePreloadSiblingsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // FORBIDHOLES 
    // --------------------------------------------------------------------- //
    /// Whether to prevent refinement of a parent tile when a child isn't done loading. When this is set to true, the tileset will guarantee that the tileset will never be rendered with holes in place of tiles that are not yet loaded, even though the tile that is rendered instead may have low resolution. When false, overall loading will be faster, but newly-visible parts of the tileset may initially be blank.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:forbidHoles = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetForbidHolesAttr() const;

    /// See GetForbidHolesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateForbidHolesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMSIMULTANEOUSTILELOADS 
    // --------------------------------------------------------------------- //
    /// The maximum number of tiles that may be loaded at once. When new parts of the tileset become visible, the tasks to load the corresponding tiles are put into a queue. This value determines how many of these tasks are processed at the same time. A higher value may cause the tiles to be loaded and rendered more quickly, at the cost of a higher network and processing load.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint cesium:maximumSimultaneousTileLoads = 20` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    UsdAttribute GetMaximumSimultaneousTileLoadsAttr() const;

    /// See GetMaximumSimultaneousTileLoadsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateMaximumSimultaneousTileLoadsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMCACHEDBYTES 
    // --------------------------------------------------------------------- //
    /// The maximum number of bytes that may be cached. Note that this value, even if 0, will never cause tiles that are needed for rendering to be unloaded. However, if the total number of loaded bytes is greater than this value, tiles will be unloaded until the total is under this number or until only required tiles remain, whichever comes first.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint64 cesium:maximumCachedBytes = 536870912` |
    /// | C++ Type | uint64_t |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt64 |
    UsdAttribute GetMaximumCachedBytesAttr() const;

    /// See GetMaximumCachedBytesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateMaximumCachedBytesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // LOADINGDESCENDANTLIMIT 
    // --------------------------------------------------------------------- //
    /// The number of loading descendants a tile should allow before deciding to render itself instead of waiting. Setting this to 0 will cause each level of detail to be loaded successively. This will increase the overall loading time, but cause additional detail to appear more gradually. Setting this to a high value like 1000 will decrease the overall time until the desired level of detail is achieved, but this high-detail representation will appear at once, as soon as it is loaded completely.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint cesium:loadingDescendantLimit = 20` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    UsdAttribute GetLoadingDescendantLimitAttr() const;

    /// See GetLoadingDescendantLimitAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateLoadingDescendantLimitAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ENABLEFRUSTUMCULLING 
    // --------------------------------------------------------------------- //
    /// Whether to cull tiles that are outside the frustum. By default this is true, meaning that tiles that are not visible with the current camera configuration will be ignored. It can be set to false, so that these tiles are still considered for loading, refinement and rendering. This will cause more tiles to be loaded, but helps to avoid holes and provides a more consistent mesh, which may be helpful for physics and shadows. Note that this will always be disabled if Use Lod Transitions is set to true.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:enableFrustumCulling = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetEnableFrustumCullingAttr() const;

    /// See GetEnableFrustumCullingAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateEnableFrustumCullingAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ENABLEFOGCULLING 
    // --------------------------------------------------------------------- //
    /// Whether to cull tiles that are occluded by fog. This does not refer to the atmospheric fog rendered by Unity, but to an internal representation of fog: Depending on the height of the camera above the ground, tiles that are far away (close to the horizon) will be culled when this flag is enabled. Note that this will always be disabled if Use Lod Transitions is set to true.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:enableFogCulling = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetEnableFogCullingAttr() const;

    /// See GetEnableFogCullingAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateEnableFogCullingAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ENFORCECULLEDSCREENSPACEERROR 
    // --------------------------------------------------------------------- //
    /// Whether a specified screen-space error should be enforced for tiles that are outside the frustum or hidden in fog. When Enable Frustum Culling and Enable Fog Culling are both true, tiles outside the view frustum or hidden in fog are effectively ignored, and so their level-of-detail doesn't matter. And in this scenario, this property is ignored. However, when either of those flags are false, these would-be-culled tiles continue to be processed, and the question arises of how to handle their level-of-detail. When this property is false, refinement terminates at these tiles, no matter what their current screen-space error. The tiles are available for physics, shadows, etc., but their level-of-detail may be very low. When set to true, these tiles are refined until they achieve the specified Culled Screen Space Error. This allows control over the minimum quality of these would-be-culled tiles.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:enforceCulledScreenSpaceError = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetEnforceCulledScreenSpaceErrorAttr() const;

    /// See GetEnforceCulledScreenSpaceErrorAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateEnforceCulledScreenSpaceErrorAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // CULLEDSCREENSPACEERROR 
    // --------------------------------------------------------------------- //
    /// The screen-space error to be enforced for tiles that are outside the frustum or hidden in fog. When Enable Frustum Culling and Enable Fog Culling are both true, tiles outside the view frustum or hidden in fog are effectively ignored, and so their level-of-detail doesn't matter. And in this scenario, this property is ignored. However, when either of those flags are false, these would-be-culled tiles continue to be processed, and the question arises of how to handle their level-of-detail. When this property is false, refinement terminates at these tiles, no matter what their current screen-space error. The tiles are available for physics, shadows, etc., but their level-of-detail may be very low. When set to true, these tiles are refined until they achieve the specified Culled Screen Space Error. This allows control over the minimum quality of these would-be-culled tiles.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `float cesium:culledScreenSpaceError = 64` |
    /// | C++ Type | float |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float |
    UsdAttribute GetCulledScreenSpaceErrorAttr() const;

    /// See GetCulledScreenSpaceErrorAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateCulledScreenSpaceErrorAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SUSPENDUPDATE 
    // --------------------------------------------------------------------- //
    /// Pauses level-of-detail and culling updates of this tileset.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:suspendUpdate = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetSuspendUpdateAttr() const;

    /// See GetSuspendUpdateAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateSuspendUpdateAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SMOOTHNORMALS 
    // --------------------------------------------------------------------- //
    /// Generate smooth normals instead of flat normals when normals are missing.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:smoothNormals = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetSmoothNormalsAttr() const;

    /// See GetSmoothNormalsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateSmoothNormalsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SHOWCREDITSONSCREEN 
    // --------------------------------------------------------------------- //
    /// Whether or not to show this tileset's credits on screen.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:showCreditsOnScreen = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetShowCreditsOnScreenAttr() const;

    /// See GetShowCreditsOnScreenAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateShowCreditsOnScreenAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // ===================================================================== //
    // Feel free to add custom code below this line, it will be preserved by 
    // the code generator. 
    //
    // Just remember to: 
    //  - Close the class declaration with }; 
    //  - Close the namespace with PXR_NAMESPACE_CLOSE_SCOPE
    //  - Close the include guard with #endif
    // ===================================================================== //
    // --(BEGIN CUSTOM CODE)--
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif
