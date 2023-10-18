from ..bindings import acquire_cesium_omniverse_interface


class CesiumInterfaceManager:
    def __init__(self):
        # Acquires the interface. Is a singleton.
        self.interface = acquire_cesium_omniverse_interface()

    def __enter__(self):
        return self.interface

    def __exit__(self, exc_type, exc_val, exc_tb):
        # We release the interface when we pull down the plugin.
        pass
