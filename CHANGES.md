# Change Log

### v0.14.0 - 2023-12-01

* Added support for custom vertex attributes. Custom attributes can be accessed by their glTF attribute name with the primvar lookup functions in MDL, e.g. `data_lookup_int` and `data_lookup_float`.

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
