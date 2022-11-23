# So you want to programmatically change a setting in Omniverse?

An optional first step is to copy the settings. The easiest way to do this is to dump them out when one of our functions in `window.py` is called. This snippet will help:

```python
import json
import carb.settings
with open("<path to desired dump file>", "w") as fh:
    fh.write(json.dumps(carb.settings.get_settings().get("/")))
```

Having these settings isn't required but it may be helpful. Once pretty printed using the JSON formatter of your choice, it can help you find file paths to help in your search, and you can take a closer look at all of the current settings.

In the case of this ticket, we needed to set a setting for the RTX renderer. A quick search of the `tokens` object gives us this path:

```
c:/users/amorris/appdata/local/ov/pkg/code-2022.2.0/kit/exts/omni.rtx.settings.core
```

Perform a grep in this folder for the menu item you wish to configure programmatically. In this case, I searched for `Normal & Tangent Space Generation Mode`. That should direct you to the file where the widget is available, and you should find the following:

```python
tbnMode = ["AUTO", "CPU", "GPU", "Force GPU"]
self._add_setting_combo("Normal & Tangent Space Generation Mode", "/rtx/hydra/TBNFrameMode", tbnMode)
```

The most important piece here is the path `/rtx/hydra/TBNFrameMode`. This refers to the path in the settings. Once you have this, programmatically changing the setting is simple:

```python
import carb.settings
carb.settings.get_settings().set("/rtx/hydra/TBNFrameMode", 1)
```

If you are unsure about what the type of the value for the setting should be, I suggest checking the JSON dump of the settings. The path `/rtx/hydra/TBNFrameMode` refers to, from root, the rtx object, the child hydra object, and followed by the TBNFrameMode property within. You can also search for the property, but beware that there may be multiple that are unrelated. For example, `TBNFrameMode` has three results total, but only one is relevant to our needs.