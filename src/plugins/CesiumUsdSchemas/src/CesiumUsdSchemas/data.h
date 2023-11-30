#ifndef CESIUMUSDSCHEMAS_GENERATED_DATA_H
#define CESIUMUSDSCHEMAS_GENERATED_DATA_H

/// \file CesiumUsdSchemas/data.h

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
// CESIUMDATAPRIM                                                             //
// -------------------------------------------------------------------------- //

/// \class CesiumData
///
/// Stores stage level data for Cesium for Omniverse/USD.
///
class CesiumData : public UsdTyped
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaKind
    static const UsdSchemaKind schemaKind = UsdSchemaKind::ConcreteTyped;

    /// Construct a CesiumData on UsdPrim \p prim .
    /// Equivalent to CesiumData::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumData(const UsdPrim& prim=UsdPrim())
        : UsdTyped(prim)
    {
    }

    /// Construct a CesiumData on the prim held by \p schemaObj .
    /// Should be preferred over CesiumData(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumData(const UsdSchemaBase& schemaObj)
        : UsdTyped(schemaObj)
    {
    }

    /// Destructor.
    CESIUMUSDSCHEMAS_API
    virtual ~CesiumData();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    CESIUMUSDSCHEMAS_API
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumData holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumData(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    CESIUMUSDSCHEMAS_API
    static CesiumData
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
    static CesiumData
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
    // PROJECTDEFAULTIONACCESSTOKEN 
    // --------------------------------------------------------------------- //
    /// DEPRECATED: A string representing the token for accessing Cesium ion assets. Will be removed in a future version.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:projectDefaultIonAccessToken = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetProjectDefaultIonAccessTokenAttr() const;

    /// See GetProjectDefaultIonAccessTokenAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateProjectDefaultIonAccessTokenAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PROJECTDEFAULTIONACCESSTOKENID 
    // --------------------------------------------------------------------- //
    /// DEPRECATED: A string representing the token ID for accessing Cesium ion assets. Will be removed in a future version.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:projectDefaultIonAccessTokenId = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetProjectDefaultIonAccessTokenIdAttr() const;

    /// See GetProjectDefaultIonAccessTokenIdAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateProjectDefaultIonAccessTokenIdAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGDISABLEMATERIALS 
    // --------------------------------------------------------------------- //
    /// Debug option that renders tilesets with materials disabled.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:disableMaterials = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugDisableMaterialsAttr() const;

    /// See GetDebugDisableMaterialsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugDisableMaterialsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGDISABLETEXTURES 
    // --------------------------------------------------------------------- //
    /// Debug option that renders tilesets with textures disabled.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:disableTextures = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugDisableTexturesAttr() const;

    /// See GetDebugDisableTexturesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugDisableTexturesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGDISABLEGEOMETRYPOOL 
    // --------------------------------------------------------------------- //
    /// Debug option that disables geometry pooling.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:disableGeometryPool = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugDisableGeometryPoolAttr() const;

    /// See GetDebugDisableGeometryPoolAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugDisableGeometryPoolAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGDISABLEMATERIALPOOL 
    // --------------------------------------------------------------------- //
    /// Debug option that disables material pooling.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:disableMaterialPool = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugDisableMaterialPoolAttr() const;

    /// See GetDebugDisableMaterialPoolAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugDisableMaterialPoolAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGDISABLETEXTUREPOOL 
    // --------------------------------------------------------------------- //
    /// Debug option that disables texture pooling.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:disableTexturePool = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugDisableTexturePoolAttr() const;

    /// See GetDebugDisableTexturePoolAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugDisableTexturePoolAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGGEOMETRYPOOLINITIALCAPACITY 
    // --------------------------------------------------------------------- //
    /// Debug option that controls the initial capacity of the geometry pool.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint64 cesium:debug:geometryPoolInitialCapacity = 2048` |
    /// | C++ Type | uint64_t |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt64 |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugGeometryPoolInitialCapacityAttr() const;

    /// See GetDebugGeometryPoolInitialCapacityAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugGeometryPoolInitialCapacityAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGMATERIALPOOLINITIALCAPACITY 
    // --------------------------------------------------------------------- //
    /// Debug option that controls the initial capacity of the material pool.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint64 cesium:debug:materialPoolInitialCapacity = 2048` |
    /// | C++ Type | uint64_t |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt64 |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugMaterialPoolInitialCapacityAttr() const;

    /// See GetDebugMaterialPoolInitialCapacityAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugMaterialPoolInitialCapacityAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGTEXTUREPOOLINITIALCAPACITY 
    // --------------------------------------------------------------------- //
    /// Debug option that controls the initial capacity of the texture pool.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint64 cesium:debug:texturePoolInitialCapacity = 2048` |
    /// | C++ Type | uint64_t |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt64 |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugTexturePoolInitialCapacityAttr() const;

    /// See GetDebugTexturePoolInitialCapacityAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugTexturePoolInitialCapacityAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGRANDOMCOLORS 
    // --------------------------------------------------------------------- //
    /// Debug option that renders tiles with random colors.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:randomColors = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugRandomColorsAttr() const;

    /// See GetDebugRandomColorsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugRandomColorsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEBUGDISABLEGEOREFERENCING 
    // --------------------------------------------------------------------- //
    /// Debug option to disable georeferencing. Tiles will be rendered in EPSG:4978 (ECEF) coordinates where (0, 0, 0) is the center of the globe, the X axis points towards the prime meridian, the Y axis points towards the 90th meridian east, and the Z axis points towards the North Pole.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:debug:disableGeoreferencing = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDebugDisableGeoreferencingAttr() const;

    /// See GetDebugDisableGeoreferencingAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDebugDisableGeoreferencingAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SELECTEDIONSERVER 
    // --------------------------------------------------------------------- //
    /// The current ion Server prim used in the Cesium for Omniverse UI.
    ///
    CESIUMUSDSCHEMAS_API
    UsdRelationship GetSelectedIonServerRel() const;

    /// See GetSelectedIonServerRel(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create
    CESIUMUSDSCHEMAS_API
    UsdRelationship CreateSelectedIonServerRel() const;

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
