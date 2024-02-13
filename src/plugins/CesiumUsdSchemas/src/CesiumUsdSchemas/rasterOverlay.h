#ifndef CESIUMUSDSCHEMAS_GENERATED_RASTEROVERLAY_H
#define CESIUMUSDSCHEMAS_GENERATED_RASTEROVERLAY_H

/// \file CesiumUsdSchemas/rasterOverlay.h

#include "pxr/pxr.h"
#include ".//api.h"
#include "pxr/usd/usd/typed.h"
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
// CESIUMRASTEROVERLAYPRIM                                                    //
// -------------------------------------------------------------------------- //

/// \class CesiumRasterOverlay
///
/// Abstract base class for prims that represent a raster overlay.
///
/// For any described attribute \em Fallback \em Value or \em Allowed \em Values below
/// that are text/tokens, the actual token is published and defined in \ref CesiumTokens.
/// So to set an attribute to the value "rightHanded", use CesiumTokens->rightHanded
/// as the value.
///
class CesiumRasterOverlay : public UsdTyped
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaKind
    static const UsdSchemaKind schemaKind = UsdSchemaKind::AbstractTyped;

    /// Construct a CesiumRasterOverlay on UsdPrim \p prim .
    /// Equivalent to CesiumRasterOverlay::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumRasterOverlay(const UsdPrim& prim=UsdPrim())
        : UsdTyped(prim)
    {
    }

    /// Construct a CesiumRasterOverlay on the prim held by \p schemaObj .
    /// Should be preferred over CesiumRasterOverlay(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumRasterOverlay(const UsdSchemaBase& schemaObj)
        : UsdTyped(schemaObj)
    {
    }

    /// Destructor.
    CESIUMUSDSCHEMAS_API
    virtual ~CesiumRasterOverlay();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    CESIUMUSDSCHEMAS_API
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumRasterOverlay holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumRasterOverlay(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    CESIUMUSDSCHEMAS_API
    static CesiumRasterOverlay
    Get(const UsdStagePtr &stage, const SdfPath &path);


protected:
    /// Returns the kind of schema this class belongs to.
    ///
    /// \sa UsdSchemaKind
    CESIUMUSDSCHEMAS_API
    UsdSchemaKind _GetSchemaKind() const override;

private:
    // needs to invoke _GetStaticTfType.
    friend class UsdSchemaRegistry;
    CESIUMUSDSCHEMAS_API
    static const TfType &_GetStaticTfType();

    static bool _IsTypedSchema();

    // override SchemaBase virtuals.
    CESIUMUSDSCHEMAS_API
    const TfType &_GetTfType() const override;

public:
    // --------------------------------------------------------------------- //
    // SHOWCREDITSONSCREEN 
    // --------------------------------------------------------------------- //
    /// Whether or not to show this raster overlay's credits on screen.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform bool cesium:showCreditsOnScreen = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetShowCreditsOnScreenAttr() const;

    /// See GetShowCreditsOnScreenAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateShowCreditsOnScreenAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ALPHA 
    // --------------------------------------------------------------------- //
    /// The alpha blending value, from 0.0 to 1.0, where 1.0 is fully opaque.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform float cesium:alpha = 1` |
    /// | C++ Type | float |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetAlphaAttr() const;

    /// See GetAlphaAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateAlphaAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // OVERLAYRENDERMETHOD 
    // --------------------------------------------------------------------- //
    /// The Cesium default material will give the raster overlay a different rendering treatment based on this selection.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform token cesium:overlayRenderMethod = "overlay"` |
    /// | C++ Type | TfToken |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Token |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    /// | \ref CesiumTokens "Allowed Values" | overlay, clip |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetOverlayRenderMethodAttr() const;

    /// See GetOverlayRenderMethodAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateOverlayRenderMethodAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMSCREENSPACEERROR 
    // --------------------------------------------------------------------- //
    /// The maximum number of pixels of error when rendering this overlay. This is used to select an appropriate level-of-detail. When this property has its default value, 2.0, it means that raster overlay images will be sized so that, when zoomed in closest, a single pixel in the raster overlay maps to approximately 2x2 pixels on the screen.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform float cesium:maximumScreenSpaceError = 2` |
    /// | C++ Type | float |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetMaximumScreenSpaceErrorAttr() const;

    /// See GetMaximumScreenSpaceErrorAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateMaximumScreenSpaceErrorAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMTEXTURESIZE 
    // --------------------------------------------------------------------- //
    /// The maximum texel size of raster overlay textures, in either direction. Images created by this overlay will be no more than this number of texels in either direction. This may result in reduced raster overlay detail in some cases.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform int cesium:maximumTextureSize = 2048` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetMaximumTextureSizeAttr() const;

    /// See GetMaximumTextureSizeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateMaximumTextureSizeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMSIMULTANEOUSTILELOADS 
    // --------------------------------------------------------------------- //
    /// The maximum number of overlay tiles that may simultaneously be in the process of loading.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform int cesium:maximumSimultaneousTileLoads = 20` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetMaximumSimultaneousTileLoadsAttr() const;

    /// See GetMaximumSimultaneousTileLoadsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateMaximumSimultaneousTileLoadsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SUBTILECACHEBYTES 
    // --------------------------------------------------------------------- //
    /// The maximum number of bytes to use to cache sub-tiles in memory. This is used by provider types, that have an underlying tiling scheme that may not align with the tiling scheme of the geometry tiles on which the raster overlay tiles are draped. Because a single sub-tile may overlap multiple geometry tiles, it is useful to cache loaded sub-tiles in memory in case they're needed again soon. This property controls the maximum size of that cache.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uniform int cesium:subTileCacheBytes = 20` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    /// | \ref SdfVariability "Variability" | SdfVariabilityUniform |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetSubTileCacheBytesAttr() const;

    /// See GetSubTileCacheBytesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateSubTileCacheBytesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

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
