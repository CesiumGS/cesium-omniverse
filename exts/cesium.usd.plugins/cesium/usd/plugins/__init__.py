import os  # noqa: F401
from pxr import Plug

pluginsRoot = os.path.join(os.path.dirname(__file__), "../../../plugins")
cesiumUsdSchemasPath = pluginsRoot + "/CesiumUsdSchemas/resources"

Plug.Registry().RegisterPlugins(cesiumUsdSchemasPath)
plugin = Plug.Registry().GetPluginWithName("CesiumUsdSchemas")
if plugin:
    plugin.Load()
else:
    print("Cannot find plugin")
