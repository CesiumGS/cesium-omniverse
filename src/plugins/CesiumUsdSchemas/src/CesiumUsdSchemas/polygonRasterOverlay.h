#ifndef CESIUMUSDSCHEMAS_GENERATED_POLYGONRASTEROVERLAY_H
#define CESIUMUSDSCHEMAS_GENERATED_POLYGONRASTEROVERLAY_H

/// \file CesiumUsdSchemas/polygonRasterOverlay.h

#include "pxr/pxr.h"
#include ".//api.h"
#include ".//rasterOverlay.h"
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
// CESIUMPOLYGONRASTEROVERLAYPRIM                                             //
// -------------------------------------------------------------------------- //

/// \class CesiumPolygonRasterOverlay
///
/// Adds a prim for representing a polygon raster overlay. Should be a child of a tileset.
///
class CesiumPolygonRasterOverlay : public CesiumRasterOverlay
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaKind
    static const UsdSchemaKind schemaKind = UsdSchemaKind::ConcreteTyped;

    /// Construct a CesiumPolygonRasterOverlay on UsdPrim \p prim .
    /// Equivalent to CesiumPolygonRasterOverlay::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumPolygonRasterOverlay(const UsdPrim& prim=UsdPrim())
        : CesiumRasterOverlay(prim)
    {
    }

    /// Construct a CesiumPolygonRasterOverlay on the prim held by \p schemaObj .
    /// Should be preferred over CesiumPolygonRasterOverlay(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumPolygonRasterOverlay(const UsdSchemaBase& schemaObj)
        : CesiumRasterOverlay(schemaObj)
    {
    }

    /// Destructor.
    CESIUMUSDSCHEMAS_API
    virtual ~CesiumPolygonRasterOverlay();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    CESIUMUSDSCHEMAS_API
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumPolygonRasterOverlay holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumPolygonRasterOverlay(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    CESIUMUSDSCHEMAS_API
    static CesiumPolygonRasterOverlay
    Get(const UsdStagePtr &stage, const SdfPath &path);

    /// Attempt to ensure a \a UsdPrim adhering to this schema at \p path
    /// is defined (according to UsdPrim::IsDefined()) on this stage.
    ///
    /// If a prim adhering to this schema at \p path is already defined on this
    /// stage, return that prim.  Otherwise author an \a SdfPrimSpec with
    /// \a specifier == \a SdfSpecifierDef and this schema's prim type name for
    /// the prim at \p path at the current EditTarget.  Author \a SdfPrimSpec s
    /// with \p specifier == \a SdfSpecifierDef and empty typeName at the
    /// current EditTarget for any nonexistent, or existing but not \a Defined
    /// ancestors.
    ///
    /// The given \a path must be an absolute prim path that does not contain
    /// any variant selections.
    ///
    /// If it is impossible to author any of the necessary PrimSpecs, (for
    /// example, in case \a path cannot map to the current UsdEditTarget's
    /// namespace) issue an error and return an invalid \a UsdPrim.
    ///
    /// Note that this method may return a defined prim whose typeName does not
    /// specify this schema class, in case a stronger typeName opinion overrides
    /// the opinion at the current EditTarget.
    ///
    CESIUMUSDSCHEMAS_API
    static CesiumPolygonRasterOverlay
    Define(const UsdStagePtr &stage, const SdfPath &path);

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
    // INVERTSELECTION 
    // --------------------------------------------------------------------- //
    /// Whether to invert the selection specified by the polygons. If this is true, only the areas outside of the polygons will be rasterized.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:invertSelection = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetInvertSelectionAttr() const;

    /// See GetInvertSelectionAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateInvertSelectionAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // CARTOGRAPHICPOLYGONBINDING 
    // --------------------------------------------------------------------- //
    /// Specifies which Cartographic Polygons to use in the raster overlay
    ///
    CESIUMUSDSCHEMAS_API
    UsdRelationship GetCartographicPolygonBindingRel() const;

    /// See GetCartographicPolygonBindingRel(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create
    CESIUMUSDSCHEMAS_API
    UsdRelationship CreateCartographicPolygonBindingRel() const;

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
