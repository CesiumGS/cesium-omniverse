import pxr.Usd
import Boost.Python

__all__ = [
    "Data",
    "Imagery",
    "TilesetAPI",
    "Tokens",
]

# TODO: Fill out these stubs since we can't seem to autogen them.
class Data(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass

class Imagery(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass

class TilesetAPI(pxr.Usd.APISchemaBase, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass

class Tokens(Boost.Python.instance):
    pass
