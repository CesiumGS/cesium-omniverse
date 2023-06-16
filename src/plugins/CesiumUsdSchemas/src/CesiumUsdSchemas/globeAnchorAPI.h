#ifndef CESIUM_GENERATED_GLOBEANCHORAPI_H
#define CESIUM_GENERATED_GLOBEANCHORAPI_H

/// \file cesium/globeAnchorAPI.h

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
// CESIUMGLOBEANCHORSCHEMAAPI                                                 //
// -------------------------------------------------------------------------- //

/// \class CesiumGlobeAnchorAPI
///
/// Adds Globe Anchoring information to a Prim for use with Cesium for Omniverse.
///
class CESIUM_API CesiumGlobeAnchorAPI : public UsdAPISchemaBase
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaType
    static const UsdSchemaType schemaType = UsdSchemaType::SingleApplyAPI;

    /// Construct a CesiumGlobeAnchorAPI on UsdPrim \p prim .
    /// Equivalent to CesiumGlobeAnchorAPI::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumGlobeAnchorAPI(const UsdPrim& prim=UsdPrim())
        : UsdAPISchemaBase(prim)
    {
    }

    /// Construct a CesiumGlobeAnchorAPI on the prim held by \p schemaObj .
    /// Should be preferred over CesiumGlobeAnchorAPI(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumGlobeAnchorAPI(const UsdSchemaBase& schemaObj)
        : UsdAPISchemaBase(schemaObj)
    {
    }

    /// Destructor.
    virtual ~CesiumGlobeAnchorAPI();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumGlobeAnchorAPI holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumGlobeAnchorAPI(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    static CesiumGlobeAnchorAPI
    Get(const UsdStagePtr &stage, const SdfPath &path);


    /// Applies this <b>single-apply</b> API schema to the given \p prim.
    /// This information is stored by adding "CesiumGlobeAnchorSchemaAPI" to the 
    /// token-valued, listOp metadata \em apiSchemas on the prim.
    /// 
    /// \return A valid CesiumGlobeAnchorAPI object is returned upon success. 
    /// An invalid (or empty) CesiumGlobeAnchorAPI object is returned upon 
    /// failure. See \ref UsdPrim::ApplyAPI() for conditions 
    /// resulting in failure. 
    /// 
    /// \sa UsdPrim::GetAppliedSchemas()
    /// \sa UsdPrim::HasAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///
    static CesiumGlobeAnchorAPI 
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
    // ADJUSTORIENTATIONFORGLOBEWHENMOVING 
    // --------------------------------------------------------------------- //
    /// Gets or sets whether to adjust the Prim's orientation based on globe curvature as the game object moves.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:anchor:adjustOrientationForGlobeWhenMoving = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetAdjustOrientationForGlobeWhenMovingAttr() const;

    /// See GetAdjustOrientationForGlobeWhenMovingAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateAdjustOrientationForGlobeWhenMovingAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // DETECTTRANSFORMCHANGES 
    // --------------------------------------------------------------------- //
    /// Gets or sets whether to automatically detect changes in the Prim's transform and update the precise globe coordinates accordingly.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:anchor:detectTransformChanges = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    UsdAttribute GetDetectTransformChangesAttr() const;

    /// See GetDetectTransformChangesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateDetectTransformChangesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // LONGITUDE 
    // --------------------------------------------------------------------- //
    /// The longitude of the anchor in degrees, in the range [-180, 180].
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:anchor:longitude = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    UsdAttribute GetLongitudeAttr() const;

    /// See GetLongitudeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateLongitudeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // LATITUDE 
    // --------------------------------------------------------------------- //
    /// The latitude of the anchor in degrees, in the range [-90, 90].
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:anchor:latitude = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    UsdAttribute GetLatitudeAttr() const;

    /// See GetLatitudeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateLatitudeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // HEIGHT 
    // --------------------------------------------------------------------- //
    /// The height of the anchor in meters above the WGS84 ellipsoid. Do not confuse this with a geoid height or height above mean sea level, which can be tens of meters higher or lower depending on where in the world the origin is located.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:anchor:height = 10` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    UsdAttribute GetHeightAttr() const;

    /// See GetHeightAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateHeightAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // POSITION 
    // --------------------------------------------------------------------- //
    /// The actual position of the globally anchored prim in the ECEF coordinate system used by Cesium for Omniverse.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double3 cesium:anchor:position = (0, 0, 0)` |
    /// | C++ Type | GfVec3d |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double3 |
    UsdAttribute GetPositionAttr() const;

    /// See GetPositionAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreatePositionAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ROTATION 
    // --------------------------------------------------------------------- //
    /// The actual rotation of the globally anchored prim oriented to the ECEF coordinate system used by Cesium for Omniverse.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double3 cesium:anchor:rotation = (0, 0, 0)` |
    /// | C++ Type | GfVec3d |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double3 |
    UsdAttribute GetRotationAttr() const;

    /// See GetRotationAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateRotationAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SCALE 
    // --------------------------------------------------------------------- //
    /// The local scaling of the prim.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double3 cesium:anchor:scale = (1, 1, 1)` |
    /// | C++ Type | GfVec3d |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double3 |
    UsdAttribute GetScaleAttr() const;

    /// See GetScaleAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateScaleAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // GEOREFERENCEBINDING 
    // --------------------------------------------------------------------- //
    /// The Georeference Origin prim used for the globe anchor calculations.
    ///
    UsdRelationship GetGeoreferenceBindingRel() const;

    /// See GetGeoreferenceBindingRel(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create
    UsdRelationship CreateGeoreferenceBindingRel() const;

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
