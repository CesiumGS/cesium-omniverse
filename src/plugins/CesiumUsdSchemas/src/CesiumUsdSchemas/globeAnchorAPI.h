#ifndef CESIUMUSDSCHEMAS_GENERATED_GLOBEANCHORAPI_H
#define CESIUMUSDSCHEMAS_GENERATED_GLOBEANCHORAPI_H

/// \file CesiumUsdSchemas/globeAnchorAPI.h

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
class CesiumGlobeAnchorAPI : public UsdAPISchemaBase
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaKind
    static const UsdSchemaKind schemaKind = UsdSchemaKind::SingleApplyAPI;

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
    CESIUMUSDSCHEMAS_API
    virtual ~CesiumGlobeAnchorAPI();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    CESIUMUSDSCHEMAS_API
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
    CESIUMUSDSCHEMAS_API
    static CesiumGlobeAnchorAPI
    Get(const UsdStagePtr &stage, const SdfPath &path);


    /// Returns true if this <b>single-apply</b> API schema can be applied to 
    /// the given \p prim. If this schema can not be a applied to the prim, 
    /// this returns false and, if provided, populates \p whyNot with the 
    /// reason it can not be applied.
    /// 
    /// Note that if CanApply returns false, that does not necessarily imply
    /// that calling Apply will fail. Callers are expected to call CanApply
    /// before calling Apply if they want to ensure that it is valid to 
    /// apply a schema.
    /// 
    /// \sa UsdPrim::GetAppliedSchemas()
    /// \sa UsdPrim::HasAPI()
    /// \sa UsdPrim::CanApplyAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///
    CESIUMUSDSCHEMAS_API
    static bool 
    CanApply(const UsdPrim &prim, std::string *whyNot=nullptr);

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
    /// \sa UsdPrim::CanApplyAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///
    CESIUMUSDSCHEMAS_API
    static CesiumGlobeAnchorAPI 
    Apply(const UsdPrim &prim);

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
    // ADJUSTORIENTATIONFORGLOBEWHENMOVING 
    // --------------------------------------------------------------------- //
    /// Gets or sets whether to adjust the Prim's orientation based on globe curvature as the game object moves.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:anchor:adjustOrientationForGlobeWhenMoving = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetAdjustOrientationForGlobeWhenMovingAttr() const;

    /// See GetAdjustOrientationForGlobeWhenMovingAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
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
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetDetectTransformChangesAttr() const;

    /// See GetDetectTransformChangesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateDetectTransformChangesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ANCHORLONGITUDE 
    // --------------------------------------------------------------------- //
    /// The longitude in degrees, in the range [-180, 180].
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:anchor:longitude = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetAnchorLongitudeAttr() const;

    /// See GetAnchorLongitudeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateAnchorLongitudeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ANCHORLATITUDE 
    // --------------------------------------------------------------------- //
    /// The latitude in degrees, in the range [-90, 90].
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:anchor:latitude = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetAnchorLatitudeAttr() const;

    /// See GetAnchorLatitudeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateAnchorLatitudeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ANCHORHEIGHT 
    // --------------------------------------------------------------------- //
    /// The height in meters above the ellipsoid.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:anchor:height = 0` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetAnchorHeightAttr() const;

    /// See GetAnchorHeightAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateAnchorHeightAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // POSITION 
    // --------------------------------------------------------------------- //
    /// The actual position of the globally anchored prim in the ECEF coordinate system.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double3 cesium:anchor:position = (0, 0, 0)` |
    /// | C++ Type | GfVec3d |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double3 |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetPositionAttr() const;

    /// See GetPositionAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreatePositionAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ROTATION 
    // --------------------------------------------------------------------- //
    /// The actual rotation of the globally anchored prim oriented to the ECEF coordinate system.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double3 cesium:anchor:rotation = (0, 0, 0)` |
    /// | C++ Type | GfVec3d |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double3 |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetRotationAttr() const;

    /// See GetRotationAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
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
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetScaleAttr() const;

    /// See GetScaleAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateScaleAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // GEOREFERENCEBINDING 
    // --------------------------------------------------------------------- //
    /// The Georeference Origin prim used for the globe anchor calculations.
    ///
    CESIUMUSDSCHEMAS_API
    UsdRelationship GetGeoreferenceBindingRel() const;

    /// See GetGeoreferenceBindingRel(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create
    CESIUMUSDSCHEMAS_API
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
