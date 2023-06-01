# Change Log

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

- Added `premake5.lua` to `cesium.omniverse` and `cesium.usd.plugins` to better support Kit templates.
- Fixed crash in the Cesium Debugging window when reloading a stage.

### v0.6.0 - 2023-05-04

- Added option to show credits on screen.
- Fixed issue where tileset traversal was happening on hidden tilesets.
- Fixed issue where tile render resources were not being released back into the Fabric mesh pool in certain cases.
- Fixed regression where the texture wrap mode was no longer clamping to edge.

### v0.5.0 - 2023-05-01

- Added material pool for better performance and to reduce texture/material loading artifacts.
- Added support for multiple viewports.
- Fixed red flashes when materials are loading.
- Fixed cyan flashes when textures are loading.
- Fixed adding imagery as base layer for existing tileset.
- Fixed Fabric types for `tilesetId` and `tileId`.
- Upgraded to cesium-native v0.23.0.

### v0.4.0 - 2023-04-03

- Fixed a crash when removing the last available access token for a tileset.
- Added search field to the asset window.
- Added placeholder token name in the create field of the token window.
- No longer printing "Error adding tileset and imagery to stage" when adding a tileset.
- Better handling of long names in the asset details panel.
- Upgraded to cesium-native v0.22.1.

### v0.3.0 - 2023-03-20

- Split the Cesium USD plugins into their own Kit extension.
- Added on-screen credits.
- Added modal dialog prompting the user to enable Fabric Scene Delegate.
- General cleanup before public release.

### v0.2.0 - 2023-03-16

- Fixed raster overlay refinement.
- Fixed a crash when removing tileset and imagery using the stage window.
- Fixed issues around loading pre-existing USD files.
- Now generating flat normals by default.
- Added property window widgets for the Cesium USD Schema attributes.
- Updated documentation.
- General cleanup.

### v0.1.0 - 2023-03-01

- The initial preview build of Cesium for Omniverse!
