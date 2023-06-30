import array
import time
import carb.events
import omni.kit.app as app
import statistics
from omni.kit.viewport.utility import get_active_viewport

FREQUENCY_IN_SECONDS: float = 0.025


class FpsSampler:
    def __init__(
        self,
    ):
        self._last_time: float = 0.0
        self._active: bool = False
        self._fps = 0.0

        self._samples = array.array("f")

        self._median: float = 0.0
        self._mean: float = 0.0
        self._low: float = 0.0
        self._high: float = 0.0

        self._viewport = get_active_viewport()

        update_stream = app.get_app().get_update_event_stream()
        self._update_frame_subscription = update_stream.create_subscription_to_pop(
            self._on_update_frame, name="cesium.performance.ON_UPDATE_FRAME"
        )

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._update_frame_subscription is not None:
            self._update_frame_subscription.unsubscribe()
            self._update_frame_subscription = None

    def start(self):
        self._last_time = time.time()
        self._active = True

    def stop(self):
        self._active = False

        if len(self._samples) > 0:
            self._mean = statistics.mean(self._samples)
            self._median = statistics.median(self._samples)
            self._low = min(self._samples)
            self._high = max(self._samples)
            self._samples = array.array("f")

    def get_mean(self):
        assert not self._active
        return self._mean

    def get_median(self):
        assert not self._active
        return self._median

    def get_low(self):
        assert not self._active
        return self._low

    def get_high(self):
        assert not self._active
        return self._high

    def get_fps(self):
        assert self._active
        return self._fps

    def _on_update_frame(self, _e: carb.events.IEvent):
        if not self._active:
            return

        current_time = time.time()
        elapsed = current_time - self._last_time
        if elapsed > FREQUENCY_IN_SECONDS:
            fps = self._viewport.fps
            self._samples.append(fps)
            self._last_time = current_time
            self._fps = fps
