import logging
import urllib.request
from io import BytesIO
from PIL import Image
import omni.ui as ui


class CesiumUriImage:
    """A wrapper around an ui.ImageProvider that provides a clean way to load images from URIs or base64 encoded data
    strings."""

    def __init__(self, src: str, **kwargs):
        self._logger = logging.getLogger(__name__)

        style_type = kwargs.pop("style_type_name_override", self.__class__.__name__)
        name = kwargs.pop("name", "")

        try:
            # This is copied to image_button.py since we seem to blow the stack if we nest any deeper when rendering.
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

        except Exception as e:
            self._logger.warning(f"Failed to load image from url: {src}")
            self._logger.error(e)

    def __getattr__(self, attr):
        return getattr(self._image, attr)

    def __setattr__(self, attr, value):
        if attr == "_image":
            super().__setattr__(attr, value)
        else:
            self._image.__setattr__(attr, value)
