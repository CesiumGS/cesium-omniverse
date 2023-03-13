import os  # noqa: F401
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

# These MUST be after the plugins are loaded.
from .extension import *  # noqa: E402 F401 F403 F405
from .utils import *  # noqa: E402 F401 F403 F405
