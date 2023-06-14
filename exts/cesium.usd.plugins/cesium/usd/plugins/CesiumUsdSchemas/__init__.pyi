import pxr.Usd
import Boost.Python

__all__ = [
    "Data",
    "Georeference",
    "Imagery",
    "TilesetAPI",
    "Tokens",
    "GlobalAnchorAPI",
]

# TODO: Fill out these stubs since we can't seem to autogen them.
class Data(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance): ...
class Georeference(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance): ...

class Imagery(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance): ...
class TilesetAPI(pxr.Usd.APISchemaBase, pxr.Usd.SchemaBase, Boost.Python.instance): ...
class Tokens(Boost.Python.instance): ...
class GlobalAnchorAPI(pxr.Usd.APISchemaBase, pxr.Usd.SchemaBase, Boost.Python.instance): ...
