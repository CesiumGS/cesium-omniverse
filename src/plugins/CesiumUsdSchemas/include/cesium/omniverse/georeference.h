#ifndef CESIUM_GENERATED_GEOREFERENCE_H
#define CESIUM_GENERATED_GEOREFERENCE_H

/// \file cesium/georeference.h

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/tf/type.h"
#include "pxr/base/vt/value.h"
#include "pxr/pxr.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/typed.h"

#include "./api.h"
#include "./tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

class SdfAssetPath;

// -------------------------------------------------------------------------- //
// CESIUMGEOREFERENCE                                                         //
// -------------------------------------------------------------------------- //

/// \class CesiumGeoreference
///
/// Represents a Cesium Georeference location.
///
class CesiumGeoreference : public UsdTyped
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaType
    static const UsdSchemaType schemaType = UsdSchemaType::AbstractTyped;

    /// Construct a CesiumGeoreference on UsdPrim \p prim .
    /// Equivalent to CesiumGeoreference::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumGeoreference(const UsdPrim& prim=UsdPrim())
        : UsdTyped(prim)
    {
    }

    /// Construct a CesiumGeoreference on the prim held by \p schemaObj .
    /// Should be preferred over CesiumGeoreference(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumGeoreference(const UsdSchemaBase& schemaObj)
        : UsdTyped(schemaObj)
    {
    }

    /// Destructor.
    CESIUM_API
    virtual ~CesiumGeoreference();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    CESIUM_API
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumGeoreference holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumGeoreference(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    CESIUM_API
    static CesiumGeoreference
    Get(const UsdStagePtr &stage, const SdfPath &path);


protected:
    /// Returns the type of schema this class belongs to.
    ///
    /// \sa UsdSchemaType
    CESIUM_API
    UsdSchemaType _GetSchemaType() const override;

private:
    // needs to invoke _GetStaticTfType.
    friend class UsdSchemaRegistry;
    CESIUM_API
    static const TfType &_GetStaticTfType();

    static bool _IsTypedSchema();

    // override SchemaBase virtuals.
    CESIUM_API
    const TfType &_GetTfType() const override;

public:
    // --------------------------------------------------------------------- //
    // LATITUDE 
    // --------------------------------------------------------------------- //
    /// A double representing the latitude in decimal degrees. Has a fallback value of 0.0.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double latitude = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUM_API
    UsdAttribute GetLatitudeAttr() const;

    /// See GetLatitudeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUM_API
    UsdAttribute CreateLatitudeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // LONGITUDE 
    // --------------------------------------------------------------------- //
    /// A double representing the longitude in decimal degrees. Has a fallback value of 0.0.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double longitude = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUM_API
    UsdAttribute GetLongitudeAttr() const;

    /// See GetLongitudeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUM_API
    UsdAttribute CreateLongitudeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // HEIGHT 
    // --------------------------------------------------------------------- //
    /// A double representing the height in meters. Has a fallback value of 0.0.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double height = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUM_API
    UsdAttribute GetHeightAttr() const;

    /// See GetHeightAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUM_API
    UsdAttribute CreateHeightAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

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
