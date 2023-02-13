#ifndef CESIUM_GENERATED_RASTEROVERLAYAPI_H
#define CESIUM_GENERATED_RASTEROVERLAYAPI_H

/// \file cesium/rasterOverlayAPI.h

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
// RASTEROVERLAYAPI                                                           //
// -------------------------------------------------------------------------- //

/// \class CesiumRasterOverlayAPI
///
/// Adds Cesium specific data to a prim for representing a raster overlay. Should be a child of a tileset.
///
class CESIUM_API CesiumRasterOverlayAPI : public UsdAPISchemaBase
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaType
    static const UsdSchemaType schemaType = UsdSchemaType::SingleApplyAPI;

    /// Construct a CesiumRasterOverlayAPI on UsdPrim \p prim .
    /// Equivalent to CesiumRasterOverlayAPI::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumRasterOverlayAPI(const UsdPrim& prim=UsdPrim())
        : UsdAPISchemaBase(prim)
    {
    }

    /// Construct a CesiumRasterOverlayAPI on the prim held by \p schemaObj .
    /// Should be preferred over CesiumRasterOverlayAPI(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumRasterOverlayAPI(const UsdSchemaBase& schemaObj)
        : UsdAPISchemaBase(schemaObj)
    {
    }

    /// Destructor.
    virtual ~CesiumRasterOverlayAPI();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumRasterOverlayAPI holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumRasterOverlayAPI(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    static CesiumRasterOverlayAPI
    Get(const UsdStagePtr &stage, const SdfPath &path);


    /// Applies this <b>single-apply</b> API schema to the given \p prim.
    /// This information is stored by adding "RasterOverlayAPI" to the 
    /// token-valued, listOp metadata \em apiSchemas on the prim.
    /// 
    /// \return A valid CesiumRasterOverlayAPI object is returned upon success. 
    /// An invalid (or empty) CesiumRasterOverlayAPI object is returned upon 
    /// failure. See \ref UsdPrim::ApplyAPI() for conditions 
    /// resulting in failure. 
    /// 
    /// \sa UsdPrim::GetAppliedSchemas()
    /// \sa UsdPrim::HasAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///

    static CesiumRasterOverlayAPI 
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
    // CESIUMRASTEROVERLAYID 
    // --------------------------------------------------------------------- //
    /// A string representing a Cesium ion imagery ID.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:rasterOverlayId = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetRasterOverlayIdAttr() const;

    /// See GetRasterOverlayIdAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateRasterOverlayIdAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // NAME 
    // --------------------------------------------------------------------- //
    /// A string representing a user-defined name for the raster overlay.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:name = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetNameAttr() const;

    /// See GetNameAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateNameAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // IONTOKENID 
    // --------------------------------------------------------------------- //
    /// A string representing the token ID for accessing this Cesium ion tileset. Overrides the default project token.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:ionTokenId = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetIonTokenIdAttr() const;

    /// See GetIonTokenIdAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateIonTokenIdAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // IONTOKEN 
    // --------------------------------------------------------------------- //
    /// A string representing the token for accessing this Cesium ion tileset. Overrides the default project token.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:ionToken = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    UsdAttribute GetIonTokenAttr() const;

    /// See GetIonTokenAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    UsdAttribute CreateIonTokenAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

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
