import os
from .extension import *
from .utils import *
from pxr import Plug

pluginsRoot = os.path.join(os.path.dirname(__file__), "../../plugins")
inMemoryAssetResolverPath = pluginsRoot + "/InMemoryAssetResolver/resources"
cesiumUsdSchemasPath = pluginsRoot + "/CesiumUsdSchemas/resources"

Plug.Registry().RegisterPlugins(inMemoryAssetResolverPath)
plugin = Plug.Registry().GetPluginWithName("InMemoryAssetResolver")
if plugin:
    plugin.Load()
else:
    print("Cannot find plugin")

Plug.Registry().RegisterPlugins(cesiumUsdSchemasPath)
plugin = Plug.Registry().GetPluginWithName("CesiumUsdSchemas")
if plugin:
    plugin.Load()
else:
    print("Cannot find plugin")
