import pxr.Usd
import Boost.Python

__all__ = [
    "Data",
    "Georeference",
    "Imagery",
    "Tileset",
    "Tokens",
]


# TODO: Fill out these stubs since we can't seem to autogen them.
class Data(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass


class Georeference(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass


class Imagery(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass


class Tileset(pxr.Usd.Typed, pxr.Usd.SchemaBase, Boost.Python.instance):
    pass


class Tokens(Boost.Python.instance):
    pass
