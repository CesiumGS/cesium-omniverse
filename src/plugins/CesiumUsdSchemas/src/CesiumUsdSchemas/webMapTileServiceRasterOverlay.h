#ifndef CESIUMUSDSCHEMAS_GENERATED_WEBMAPTILESERVICERASTEROVERLAY_H
#define CESIUMUSDSCHEMAS_GENERATED_WEBMAPTILESERVICERASTEROVERLAY_H

/// \file CesiumUsdSchemas/webMapTileServiceRasterOverlay.h

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
// CESIUMWEBMAPTILESERVICERASTEROVERLAYPRIM                                   //
// -------------------------------------------------------------------------- //

/// \class CesiumWebMapTileServiceRasterOverlay
///
/// Adds a prim for representing a Web Map Tile Service (WMTS) raster overlay.
///
class CesiumWebMapTileServiceRasterOverlay : public CesiumRasterOverlay
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaKind
    static const UsdSchemaKind schemaKind = UsdSchemaKind::ConcreteTyped;

    /// Construct a CesiumWebMapTileServiceRasterOverlay on UsdPrim \p prim .
    /// Equivalent to CesiumWebMapTileServiceRasterOverlay::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit CesiumWebMapTileServiceRasterOverlay(const UsdPrim& prim=UsdPrim())
        : CesiumRasterOverlay(prim)
    {
    }

    /// Construct a CesiumWebMapTileServiceRasterOverlay on the prim held by \p schemaObj .
    /// Should be preferred over CesiumWebMapTileServiceRasterOverlay(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit CesiumWebMapTileServiceRasterOverlay(const UsdSchemaBase& schemaObj)
        : CesiumRasterOverlay(schemaObj)
    {
    }

    /// Destructor.
    CESIUMUSDSCHEMAS_API
    virtual ~CesiumWebMapTileServiceRasterOverlay();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    CESIUMUSDSCHEMAS_API
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a CesiumWebMapTileServiceRasterOverlay holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// CesiumWebMapTileServiceRasterOverlay(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    CESIUMUSDSCHEMAS_API
    static CesiumWebMapTileServiceRasterOverlay
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
    static CesiumWebMapTileServiceRasterOverlay
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
    // URL 
    // --------------------------------------------------------------------- //
    /// The base url of the Web Map Tile Service (WMTS).
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:url = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetUrlAttr() const;

    /// See GetUrlAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateUrlAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // LAYER 
    // --------------------------------------------------------------------- //
    /// Layer name.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:layer = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetLayerAttr() const;

    /// See GetLayerAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateLayerAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // STYLE 
    // --------------------------------------------------------------------- //
    /// Style.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:style = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetStyleAttr() const;

    /// See GetStyleAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateStyleAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // FORMAT 
    // --------------------------------------------------------------------- //
    /// Format.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:format = "image/jpeg"` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetFormatAttr() const;

    /// See GetFormatAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateFormatAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // TILEMATRIXSETID 
    // --------------------------------------------------------------------- //
    /// Tile Matrix Set ID
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:tileMatrixSetId = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetTileMatrixSetIdAttr() const;

    /// See GetTileMatrixSetIdAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateTileMatrixSetIdAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SPECIFYTILEMATRIXSETLABELS 
    // --------------------------------------------------------------------- //
    /// True to specify tile matrix set labels manually, or false to automatically determine from level and prefix.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:specifyTileMatrixSetLabels = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetSpecifyTileMatrixSetLabelsAttr() const;

    /// See GetSpecifyTileMatrixSetLabelsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateSpecifyTileMatrixSetLabelsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // TILEMATRIXSETLABELPREFIX 
    // --------------------------------------------------------------------- //
    /// Prefix for tile matrix set labels. For instance, setting "EPSG:4326:" as prefix generates label list ["EPSG:4326:0", "EPSG:4326:1", "EPSG:4326:2", ...]
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:tileMatrixSetLabelPrefix = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetTileMatrixSetLabelPrefixAttr() const;

    /// See GetTileMatrixSetLabelPrefixAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateTileMatrixSetLabelPrefixAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // TILEMATRIXSETLABELS 
    // --------------------------------------------------------------------- //
    /// Tile Matrix Set Labels.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `string cesium:tileMatrixSetLabels = ""` |
    /// | C++ Type | std::string |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->String |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetTileMatrixSetLabelsAttr() const;

    /// See GetTileMatrixSetLabelsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateTileMatrixSetLabelsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // USEWEBMERCATORPROJECTION 
    // --------------------------------------------------------------------- //
    /// False to use geographic projection, true to use webmercator projection. For instance, EPSG:4326 uses geographic and EPSG:3857 uses webmercator.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:useWebMercatorProjection = 1` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetUseWebMercatorProjectionAttr() const;

    /// See GetUseWebMercatorProjectionAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateUseWebMercatorProjectionAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SPECIFYTILINGSCHEME 
    // --------------------------------------------------------------------- //
    /// True to specify quadtree tiling scheme according to projection and bounding rectangle, or false to automatically determine from projection.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:specifyTilingScheme = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetSpecifyTilingSchemeAttr() const;

    /// See GetSpecifyTilingSchemeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateSpecifyTilingSchemeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ROOTTILESX 
    // --------------------------------------------------------------------- //
    /// Tile number corresponding to TileCol, also known as TileMatrixWidth
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `int cesium:rootTilesX = 1` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetRootTilesXAttr() const;

    /// See GetRootTilesXAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateRootTilesXAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // ROOTTILESY 
    // --------------------------------------------------------------------- //
    /// Tile number corresponding to TileRow, also known as TileMatrixHeight
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `int cesium:rootTilesY = 1` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetRootTilesYAttr() const;

    /// See GetRootTilesYAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateRootTilesYAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // WEST 
    // --------------------------------------------------------------------- //
    /// The longitude of the west boundary on globe in degrees, in the range [-180, 180]
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:west = -180` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetWestAttr() const;

    /// See GetWestAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateWestAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // EAST 
    // --------------------------------------------------------------------- //
    /// The longitude of the east boundary on globe in degrees, in the range [-180, 180]
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:east = 180` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetEastAttr() const;

    /// See GetEastAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateEastAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SOUTH 
    // --------------------------------------------------------------------- //
    /// The longitude of the south boundary on globe in degrees, in the range [-90, 90]
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:south = -90` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetSouthAttr() const;

    /// See GetSouthAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateSouthAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // NORTH 
    // --------------------------------------------------------------------- //
    /// The longitude of the north boundary on globe in degrees, in the range [-90, 90]
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `double cesium:north = 90` |
    /// | C++ Type | double |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Double |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetNorthAttr() const;

    /// See GetNorthAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateNorthAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // SPECIFYZOOMLEVELS 
    // --------------------------------------------------------------------- //
    /// True to directly specify minum and maximum zoom levels available from the server, or false to automatically determine the minimum and maximum zoom levels from the server's tilemapresource.xml file.
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool cesium:specifyZoomLevels = 0` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetSpecifyZoomLevelsAttr() const;

    /// See GetSpecifyZoomLevelsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateSpecifyZoomLevelsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MINIMUMZOOMLEVEL 
    // --------------------------------------------------------------------- //
    /// Minimum zoom level
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `int cesium:minimumZoomLevel = 0` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetMinimumZoomLevelAttr() const;

    /// See GetMinimumZoomLevelAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateMinimumZoomLevelAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // MAXIMUMZOOMLEVEL 
    // --------------------------------------------------------------------- //
    /// Maximum zoom level
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `int cesium:maximumZoomLevel = 25` |
    /// | C++ Type | int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Int |
    CESIUMUSDSCHEMAS_API
    UsdAttribute GetMaximumZoomLevelAttr() const;

    /// See GetMaximumZoomLevelAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    CESIUMUSDSCHEMAS_API
    UsdAttribute CreateMaximumZoomLevelAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

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
