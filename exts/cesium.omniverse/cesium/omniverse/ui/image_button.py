import urllib.request
from io import BytesIO
from PIL import Image
import omni.ui as ui


class CesiumImageButton:
    """A button with an image from a URL or base64 encoded string. Based off of Nvidia's ButtonWithProvider sample."""

    def __init__(self, src: str, button_type=ui.Button, **kwargs):
        style_type = kwargs.pop("style_type_name_override", self.__class__.__name__)
        name = kwargs.pop("name", "")

        with ui.ZStack(height=0, width=0):
            # This is copied from uri_image.py since we seem to blow the stack if we nest any deeper when rendering.
            data = urllib.request.urlopen(src).read()
            img_data = BytesIO(data)
            image = Image.open(img_data)
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            pixels = list(image.getdata())
            provider = ui.ByteImageProvider()
            provider.set_bytes_data(pixels, [image.size[0], image.size[1]])

            self._image = ui.ImageWithProvider(
                provider,
                width=image.size[0],
                height=image.size[1],
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                style_type_name_override=style_type,
                name=name,
            )

            self._button = button_type(
                text=" ",  # Workaround Buttons without text do not expand vertically
                style_type_name_override=style_type,
                name=name,
                style={"background_color": ui.color.transparent, "color": ui.color.transparent},
                **kwargs,
            )

    def get_image(self):
        return self._image

    def __getattr__(self, attr):
        return getattr(self._button, attr)

    def __setattr__(self, attr, value):
        if attr == "_image" or attr == "_button":
            super().__setattr__(attr, value)
        else:
            self._button.__setattr__(attr, value)
