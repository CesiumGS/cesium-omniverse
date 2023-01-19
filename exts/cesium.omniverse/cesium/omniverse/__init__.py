import os
from .extension import *
from .utils import *
from pxr import Plug

pluginsRoot = os.path.join(os.path.dirname(__file__), "../../plugins")
inMemoryAssetResolverPath = pluginsRoot + "/InMemoryAssetResolver/resources"

Plug.Registry().RegisterPlugins(inMemoryAssetResolverPath)
plugin = Plug.Registry().GetPluginWithName("InMemoryAssetResolver")
if plugin:
    plugin.Load()
else:
    print("Cannot find plugin")
