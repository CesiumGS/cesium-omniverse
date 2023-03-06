import logging
import omni.kit.app as app
import omni.ui as ui
import omni.kit.pipapi
import urllib.request
import webbrowser
from pathlib import Path
from io import BytesIO
from PIL import Image
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface
from .styles import CesiumOmniverseUiStyles


class CesiumOmniverseCreditsWindow(ui.Window):
    WINDOW_NAME = "Data Attribution"

    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        super().__init__(CesiumOmniverseCreditsWindow.WINDOW_NAME, **kwargs)

        manager = app.get_app().get_extension_manager()
        ext_id = manager.get_extension_id_by_module("cesium.omniverse")

        self._cesium_omniverse_interface = cesium_omniverse_interface
        self._logger = logging.getLogger(__name__)
        self._images_path = Path(manager.get_extension_path(ext_id)).joinpath("images")

        self.height = 500
        self.width = 400

        self.padding_x = 12
        self.padding_y = 12

        self._credits: List[(str, bool)] = self._cesium_omniverse_interface.get_credits()

        self.frame.set_build_fn(self._build_ui)

    def __del__(self):
        self.destroy()

    def destroy(self):
        pass

        super().destroy()

    def _parse_element(self, element, link: Optional[str] = None):
        tag = element.tag
        if tag == "html" or tag == "body":
            for child in element.iterchildren():
                self._parse_element(child, link)
        elif tag == "a":
            # TODO: We probably need to do some sanitization of the href.
            link = element.attrib["href"]
            text = "".join(element.itertext())

            if text != "":
                ui.Button(text, height=0, width=0, clicked_fn=lambda: webbrowser.open(link))
            for child in element.iterchildren():
                self._parse_element(child, link)
        elif tag == "img":
            src = element.attrib["src"]
            try:
                data = urllib.request.urlopen(src).read()
                img_data = BytesIO(data)
                image = Image.open(img_data)
                if image.mode != "RGBA":
                    image = image.convert("RGBA")
                pixels = list(image.getdata())
                provider = ui.ByteImageProvider()
                provider.set_bytes_data(pixels, [image.size[0], image.size[1]])
                ui.ImageWithProvider(
                    provider,
                    width=image.size[0],
                    height=image.size[1],
                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                )
            except Exception as e:
                self._logger.warning(f"Failed to load image from url: {src}")
                self._logger.error(e)

            if link is not None:
                link_title = element.attrib["alt"] if "alt" in element.attrib else link
                ui.Button(link_title, clicked_fn=lambda: webbrowser.open(link), height=0, width=0)
        elif tag == "span" or tag == "div":
            for child in element.iterchildren():
                self._parse_element(child, link)

            # Sometimes divs or spans have text.
            text = "".join(element.itertext())
            if text:
                ui.Label(text, height=0, word_wrap=True)
        else:
            text = "".join(element.itertext())
            if link is None:
                ui.Label(text, height=0, word_wrap=True)
            else:
                ui.Button(text, clicked_fn=lambda: webbrowser.open(link), height=0, width=0)

    def _build_ui(self):
        with ui.VStack(spacing=5):
            ui.Label("Data Provided By:", height=0, style=CesiumOmniverseUiStyles.attribution_header_style)

            try:
                # This installs lxml and base64 which are needed for credit display.
                omni.kit.pipapi.install("lxml==4.9.2")
                from lxml import etree

                parser = etree.HTMLParser()
                for credit, _ in self._credits:
                    if credit == "":
                        continue

                    if credit[0] == "<":
                        try:
                            doc = etree.fromstring(credit, parser)
                            self._parse_element(doc)
                            continue
                        except etree.XMLSyntaxError as err:
                            self._logger.info(err)

                    ui.Label(credit, height=0, word_wrap=True)
            except Exception as e:
                self._logger.error(e)
                self._logger.warning("Performing credits fallback.")

                for credit, _ in self._credits:
                    ui.Label(credit, height=0, word_wrap=True)
