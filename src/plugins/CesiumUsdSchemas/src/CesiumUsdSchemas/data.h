#ifndef CESIUM_GENERATED_DATA_H
#define CESIUM_GENERATED_DATA_H

/// \file cesium/data.h

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
// CESIUMDATA                                                                 //
// -------------------------------------------------------------------------- //

/// \class CesiumData
///
/// Stores stage level data for Cesium for Omniverse/USD.
///
class CESIUM_API CesiumData : public UsdTyped
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaType
    static const UsdSchemaType schemaType = UsdSchemaType::ConcreteTyped;

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
    virtual ~CesiumData();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
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
    static CesiumData
    Define(const UsdStagePtr &stage, const SdfPath &path);

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
    // DEFAULTPROJECTTOKENID 
    // --------------------------------------------------------------------- //
    /// A string representing the token ID for accessing Cesium ion tilesets.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:defaultProjectTokenId = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetDefaultProjectTokenIdAttr() const;

    /// See GetDefaultProjectTokenIdAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateDefaultProjectTokenIdAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DEFAULTPROJECTTOKEN 
    // --------------------------------------------------------------------- //
    /// A string representing a token for accessing Cesium ion tilesets.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:defaultProjectToken = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetDefaultProjectTokenAttr() const;

    /// See GetDefaultProjectTokenAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateDefaultProjectTokenAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // GEOREFERENCEORIGIN 
    // --------------------------------------------------------------------- //
    /// Specifies a Georeference origin point for Cesium in Longitude, Latitude, and Height
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double3 cesium:georeferenceOrigin = (0, 0, 0)` |
    /// | C++ Type | GfVec3d |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double3 |
    UsdAttribute GetGeoreferenceOriginAttr() const;

    /// See GetGeoreferenceOriginAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateGeoreferenceOriginAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

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
