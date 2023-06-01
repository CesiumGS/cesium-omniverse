from carb.events import type_from_string

# Event base path. Do not use directly outside of events.py
_EVENT_BASE = "cesium.omniverse.event"

# Signals the credits have changed. Currently, this only triggers when the displayed
# credits are changed. It is possible for the credit payload that shows under
# the "Data Attribution" button to change, but this event will not fire for that.
EVENT_CREDITS_CHANGED = type_from_string(f"{_EVENT_BASE}.viewport.CREDITS_CHANGED")
