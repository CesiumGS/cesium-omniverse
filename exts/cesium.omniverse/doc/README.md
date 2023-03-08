# Cesium for Omniverse

Cesium for Omniverse enables building 3D geospatial applications and experiences with 3D Tiles and open standards in NVIDIA Omnniverse, a real-time 3D graphics collaboration development platform. Cesium for Omniverse is an extension for Omniverse's Kit-based applications such as USD Composer (formerly Create), a reference application for large-scale world building and advanced 3D scene composition of Universal Scene Description (USD)-based workflows, and Omniverse Code, an integrated development environment (IDE) for developers. Cesium for Omniverse enables developers to create geospatial and enterprise metaverse, simulations, digital twins, and other real-world applications in precise geospatial context and stream massive real-word 3D content.

By combining a high-accuracy full-scale WGS84 globe, open APIs and open standards for spatial indexing such as 3D Tiles, and cloud-based real-world content from [Cesium ion](https://cesium.com/cesium-ion) with Omniverse, this extension enables rich 3D geospatial workflows and applications in Omniverse, which adds real-time ray tracing and AI-powered analytics.

## Activating the Extension

Cesium for Omniverse uses preview features built into Omniverse Create 2023.3.3 and later.  The following steps must be followed to enable these features and ensure correct functionality of the extension:

1. Enable the Cesium for Omniverse extension and tick the Autoload checkbox
2. Restart Omniverse Create to ensure the extension loads correctly
3. Open Edit > Preferences > Rendering and tick Enable Fabric delegate
4. Reload or create a new stage
5. The extension is now ready to be used. Follow our tutorials at https://cesium.com/learn/omniverse/

## Known Issues and limitations

The Enable Fabric delegate setting is a preview feature Cesium for Omniverse utilises to ensure optimal loading and rendering performance.  Some features in Omniverse Create may not function correctly with this setting enabled.  Known issues include:

- Dynamic skies are not currently compatible with the Enable Fabric delegate setting
- The Enable Fabric delegate setting defaults to unticked when you restart Omniverse Create.  Be sure to tick this setting each time you open Omniverse Create when using Cesium for Omniverse

You can safely toggle this setting between enabled and disabled without impacting the content of your stage, however Cesium for Omniverse data will only render when this setting is enabled.  A reload of your stage is required each time you change this setting.