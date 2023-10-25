import carb.settings
import omni.ui as ui
import omni.kit.window.file


class CesiumFabricModal(ui.Window):
    def __init__(self):
        window_flags = ui.WINDOW_FLAGS_NO_RESIZE
        window_flags |= ui.WINDOW_FLAGS_NO_SCROLLBAR
        window_flags |= ui.WINDOW_FLAGS_MODAL
        window_flags |= ui.WINDOW_FLAGS_NO_CLOSE
        window_flags |= ui.WINDOW_FLAGS_NO_COLLAPSE

        super().__init__("Enable Fabric", height=200, width=300, flags=window_flags)

        self.frame.set_build_fn(self._build_fn)

    def _on_yes_click(self):
        carb.settings.get_settings().set_bool("/app/useFabricSceneDelegate", True)
        carb.settings.get_settings().set_bool("/app/usdrt/scene_delegate/enableProxyCubes", False)
        carb.settings.get_settings().set_bool("/app/usdrt/scene_delegate/geometryStreaming/enabled", False)
        carb.settings.get_settings().set_bool("/omnihydra/parallelHydraSprimSync", False)
        omni.kit.window.file.new()
        self.visible = False

    def _on_no_click(self):
        self.visible = False

    def _build_fn(self):
        with ui.VStack(height=0, spacing=10):
            ui.Label(
                "The Omniverse Fabric Scene Delegate is currently disabled. Cesium for Omniverse requires the "
                "Fabric Scene Delegate to be enabled to function.",
                word_wrap=True,
            )
            ui.Label("Would you like to enable the Fabric Scene Delegate and create a new stage?", word_wrap=True)
            ui.Button("Yes", clicked_fn=self._on_yes_click)
            ui.Button("No", clicked_fn=self._on_no_click)
