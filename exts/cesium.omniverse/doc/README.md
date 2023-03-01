# Cesium for Omniverse
The Cesium for Omniverse extension enables the 3D geospatial ecosystem in NVIDIA Omniverse Kit-based Applications. Cesium for Omniverse connects the real world with the metaverse by empowering software developers to build digital twins using accurate, massive, real-world 3D geospatial data by combining a high-accuracy full-scale WGS84 globe, open APIs and open standards for spatial indexing like 3D Tiles, and cloud-based real-world 3D content with the power of Omniverse.

### Activating the Extension
Cesium for Omniverse uses preview features built into Omniverse Create 2023.3.3 and later.  The following steps must be followed to enable these features and ensure correct functionality of the extension:

1) Enable the Cesium for Omniverse extension and tick the Autoload checkbox
2) Restart Omniverse Create to ensure the extension loads correctly
3) Open Edit > Preferences > Rendering and tick Enable Fabric delegate
4) Reload or create a new stage
5) The extension is now ready to be used.  Follow our tutorials at <enter link here>

### Known Issues and limitations
The Enable Fabric delegate setting is a preview feature Cesium for Omniverse utilises to ensure optimal loading and rendering performance.  Some features in Omniverse Create may not function correctly with this setting enabled.  Known issues include:

- Dynamic skies are not currently compatible with the Enable Fabric delegate setting
- The Enable Fabric delegate setting defaults to unticked when you restart Omniverse Create.  Be sure to tick this setting each time you open Omniverse Create when using Cesium for Omniverse

You can safely toggle this setting between enabled and disabled without impacting the content of your stage, however Cesium for Omniverse data will only render when this setting is enabled.  A reload of your stage is required each time you change this setting.

