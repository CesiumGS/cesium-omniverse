# Change Log

### v0.24.0 - 2024-11-01

* Fixed bug where smooth normals setting would crash.

### v0.23.0 - 2024-10-01

* Fixed bug where tilesets and raster overlays were not being passed the correct custom ellipsoid.
* Fixed rendering base color textures that contain an alpha channel.

### v0.22.0 - 2024-09-03

* Cesium for Omniverse now supports using non-WGS84 ellipsoids.
  * A `CesiumEllipsoidPrim` should be specified on the `ellipsoidBinding` field of a `CesiumReferencePrim`.

### v0.21.0 - 2024-06-03

* Added `pointSize` attribute to `CesiumTilesetPrim` for controlling the size of points.
* Added read-only attribute `ecefToUsdTransform` to `CesiumGeoreferencePrim`. Previously this was stored in `/CesiumSession` which has since been removed.
* Fixed crash when updating globe anchor when georeferencing is disabled.
* Fixed point cloud styling.

### v0.20.0 - 2024-05-01

* Fix missing button for adding imagery from Asset UI.
* Updated cesium-native which includes a bug fix for reading GLB files with extra padding bytes.

### v0.19.0 - 2024-04-01

* Added scrollbar to main window UI.
* Fixed issue when loading tilesets with Cesium ion Self-Hosted in developer mode.

### v0.18.0 - 2024-03-01

* **Breaking change:** removed deprecated properties `projectDefaultIonAccessToken` and `projectDefaultIonAccessToken` from `CesiumDataPrim`. `CesiumIonServerPrim` should be used instead.
* Improved tile streaming performance by 35% by switching to UrlAssetAccessor from vsgCs.
* Added support for disk tile caching which improves streaming performance by 50% when reloading the same scene.
* Added support for Web Map Service (WMS) raster overlays.
* Added support for Tile Map Service (TMS) raster overlays.
* Added support for Web Map Tile Service (WMTS) raster overlays.
* Added raster overlay options: `maximumScreenSpaceError`, `maximumTextureSize`, `maximumSimultaneousTileLoads`, `subTileCacheBytes`.
* Added ability to bypass downloading of tiles clipped by a cartographic polygon raster overlay.
* Added support for globe anchors on non-georeferenced tilesets.
* Fixed crash when disabling and re-enabling the extension.
* Fixed crash when setting certain `/Cesium` debug options at runtime.
* Fixed crash when updating tilesets shader inputs.
* Fixed crash when removing USD prims in certain order.
* Fixed issue where Cesium ion session would not resume on reload.
* Fixed issue where save stage dialog would appear when reloading Fabric stage at startup.
* Fixed issue where zooming to tileset extents would not work correctly with non-identity transformation.
* Fixed issue where globe anchors didn't work with `xformOp:orient`.
* The movie capture tool now waits for tilesets to complete loading before it captures a frame.

### v0.17.0 - 2024-02-01

* **Breaking changes for globe anchors:**
  * Removed `anchor_xform_at_path`. Globe anchors can now be created directly in USD.
  * Split `cesium:anchor:geographicCoordinates` into separate properties: `cesium:anchor:latitude`, `cesium:anchor:longitude`, `cesium:anchor:height`.
  * Globe anchors no longer add a `transform:cesium` op to the attached prim. Instead the `translate`, `rotateXYZ`, and `scale` ops are modified directly.
  * Removed `cesium:anchor:rotation` and `cesium:anchor:scale`. Instead, use `UsdGeom.XformCommonAPI` to modify the globe anchor's local rotation and scale.
  * Globe anchors now use the scene's default georeference if `cesium:georeferenceBinding` is empty.
  * For migrating existing USD files, see https://github.com/CesiumGS/cesium-omniverse-samples/pull/13
* **Breaking changes for imagery layers:**
  * `CesiumImagery` was renamed to `CesiumRasterOverlay` and is now an abstract class. To create ion raster overlays, use `CesiumIonRasterOverlay`.
  * MDL changes: `cesium_imagery_layer_float4` was renamed to `cesium_raster_overlay_float4` and `imagery_layer_index` was renamed to `raster_overlay_index`.
  * ion raster overlays now use the scene's default ion server if `cesium:ionServerBinding` is empty.
* **Breaking change for tilesets:**
  * Tilesets must now reference raster overlays with `cesium:rasterOverlayBinding`.
  * Tilesets now use the scene's default georeference if `cesium:georeferenceBinding` is empty.
  * Tilesets now uses the scene's default ion server if `cesium:ionServerBinding` is empty.
* Added support for polygon-based clipping with `CesiumPolygonRasterOverlay`.
* Added ability for multiple tilesets referencing the same raster overlay.
* Added ability to reorder raster overlays in UI.
* Added context menu options for adding raster overlays to tilesets.
* Fixed multiple globe anchor related issues.
* Fixed excessive property warnings when using custom materials.
* Fixed adding raster overlays to selected tileset in the Add Assets UI.
* Fixed loading 3D Tiles 1.1 implicit tilesets.

### v0.16.0 - 2024-01-02

* Fixed issue where the current ion session would be signed out on reload.
* Fixed crash in Cesium Debugging window.

### v0.15.0 - 2023-12-14

* Added support for multiple Cesium ion servers by creating `CesiumIonServerPrim` prims.

### v0.14.0 - 2023-12-01

* Added support for `EXT_structural_metadata`. Property values can be accessed in material graph with the `cesium_property` nodes.
* Added support for `EXT_mesh_features`. Feature ID values can be accessed in material graph with the `cesium_feature_id_int` node.
* Added support for custom glTF vertex attributes. Attribute values can be accessed in material graph with the `data_lookup` nodes.
* Added support for changing a tileset's imagery layer dynamically in material graph.

### v0.13.0 - 2023-11-01

* Changing certain tileset properties no longer triggers a tileset reload.
* Added support for `displayColor` and `displayOpacity` for tileset prims.
* Fixed rendering point clouds with alpha values.

### v0.12.1 - 2023-10-26

* Fixed version numbers.

### v0.12.0 - 2023-10-25

* Added a quick add button for Google Photorealistic 3D Tiles through ion.
* Added support for globe anchors.
* Added support for multiple imagery layers.
* Added alpha property to imagery layers.
* Added support for reading textures and imagery layers in MDL.
* Added Cesium for Omniverse Python API, see the `cesium.omniverse.api` module.
* Fixed debug colors not working for tiles with vertex colors.
* Fixed hangs when loading tilesets by setting `omnihydra.parallelHydraSprimSync` to `false`.
* Basis Universal textures are now decoded to the native BCn texture format instead of RGBA8 in Kit 105.1 and above.

### v0.11.0 - 2023-10-02

* **Breaking change:** Cesium for Omniverse now requires Kit 105.1 or above (USD Composer 2023.2.0 or above).
* Reduced the number of materials created when loading un-textured tilesets.
* Added debug option `cesium:debug:disableGeoreferencing` to `CesiumDataPrim` to disable georeferencing and view tilesets in ECEF coordinates.
* Improvements to C++ testing infrastructure.

### v0.10.0 - 2023-09-01

* Improved error message if fetching tileset fails.
* Added basic point cloud support.
* Fixed loading extension in Omniverse Code 2023.1.1.
* Fixed crashes when reloading tileset.
* Fixed memory leak when removing tileset mid-load.
* Fixed several other bugs related to removing tilesets mid-load.
* Upgraded to cesium-native v0.26.0.

### v0.9.0 - 2023-08-01

* **Breaking change:** `CesiumTilesetPrim` now inherits from `UsdGeomGprim` instead of `UsdGeomBoundable`.
* Improved texture loading performance by moving texture loading to a worker thread.
* Improved performance when refining with parent tile's imagery by sharing the same texture instead of duplicating it.
* Added support for assigning materials to a tileset.
* Improved styling of credits.
* Visually enable/disable top bar buttons based on sign-in status.
* Fixed bug where not all Cesium windows would not appear in Windows menu.

### v0.8.0 - 2023-07-03

* **Breaking change:** Cesium for Omniverse now requires Kit 105 or above (USD Composer 2023.1.0 or above).
* **Breaking change:** broke out georeference attributes from `CesiumDataPrim` into dedicated `CesiumGeoreferencePrim` class.
* **Breaking change:** `CesiumTilesetPrim` is now a concrete type that inherits from `UsdGeomBoundable`.
* **Breaking change:** `CesiumTilesetPrim` now has an explicit binding to a `CesiumGeoreferencePrim`.
* **Breaking change:** default values for attributes are no longer written out when saved as `.usd` files.
* Added ability to zoom to extents on tilesets.
* Added vertex color support.
* Added `cesium.omniverse.TILESET_LOADED` Carbonite event.
* Added more statistics to the Cesium Debugging window.
* Fixed holes when camera is moving.
* Fixed orphaned tiles.
* Fixed credit parsing issue.
* Improved performance when refining with parent tile's imagery.
* Improved performance when creating Fabric geometry.
* Switched base material to `gltf/pbr.mdl`.

### v0.7.0 - 2023-06-01

* Set better default values when loading glTFs with the `KHR_materials_unlit` extension. This improves the visual quality of Google 3D Tiles.
* Improved installation process by forcing application reload when Cesium for Omniverse is first enabled.
* Changed material loading color from red to black.
* Added `/CesiumSession` prim for storing ephemeral state in the Session Layer, including `ecefToUsdTransform`.
* Fixed credits not appearing on all viewports.
* Improved readability of debug statistics.
* Integrated Cesium Native's performance tracing utility.
* Updated to Cesium Native 0.24.0 which adds support for 3D Tiles 1.1 implicit tiling.

### v0.6.2 - 2023-05-19

* Added more rendering statistics to the Cesium Debugging window.
* Added debug options to the top-level `Cesium` prim.
* Fixed issue where `cesium:enableFrustumCulling` wasn't appearing in the USD schema UI.
* Fixed issue where some Fabric shader node prims were not being deleted.

### v0.6.1 - 2023-05-11

* Added `premake5.lua` to `cesium.omniverse` and `cesium.usd.plugins` to better support Kit templates.
* Fixed crash in the Cesium Debugging window when reloading a stage.

### v0.6.0 - 2023-05-04

* Added option to show credits on screen.
* Fixed issue where tileset traversal was happening on hidden tilesets.
* Fixed issue where tile render resources were not being released back into the Fabric mesh pool in certain cases.
* Fixed regression where the texture wrap mode was no longer clamping to edge.

### v0.5.0 - 2023-05-01

* Added material pool for better performance and to reduce texture/material loading artifacts.
* Added support for multiple viewports.
* Fixed red flashes when materials are loading.
* Fixed cyan flashes when textures are loading.
* Fixed adding imagery as base layer for existing tileset.
* Fixed Fabric types for `tilesetId` and `tileId`.
* Upgraded to cesium-native v0.23.0.

### v0.4.0 - 2023-04-03

* Fixed a crash when removing the last available access token for a tileset.
* Added search field to the asset window.
* Added placeholder token name in the create field of the token window.
* No longer printing "Error adding tileset and imagery to stage" when adding a tileset.
* Better handling of long names in the asset details panel.
* Upgraded to cesium-native v0.22.1.

### v0.3.0 - 2023-03-20

* Split the Cesium USD plugins into their own Kit extension.
* Added on-screen credits.
* Added modal dialog prompting the user to enable Fabric Scene Delegate.
* General cleanup before public release.

### v0.2.0 - 2023-03-16

* Fixed raster overlay refinement.
* Fixed a crash when removing tileset and imagery using the stage window.
* Fixed issues around loading pre-existing USD files.
* Now generating flat normals by default.
* Added property window widgets for the Cesium USD Schema attributes.
* Updated documentation.
* General cleanup.

### v0.1.0 - 2023-03-01

* The initial preview build of Cesium for Omniverse!
