import logging
import omni.kit.pipapi
import omni.ui as ui
import webbrowser
from typing import Optional, List, Tuple
from .uri_image import CesiumUriImage
from .image_button import CesiumImageButton


class CesiumCreditsParser:
    """Takes in a credits array and outputs the elements necessary to show the credits.
    Should be embedded in a VStack or HStack."""

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
            if link is None:
                self._logger.warning("image")
                CesiumUriImage(src=src)
            else:
                CesiumImageButton(src=src, padding=4, clicked_fn=lambda: webbrowser.open(link))
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

    # There is a builtin name called credits, which is why this argument is called asset_credits.
    def __init__(self, asset_credits: List[Tuple[str, bool]], should_show_on_screen: bool, perform_fallback=False):
        self._logger = logging.getLogger(__name__)

        try:
            omni.kit.pipapi.install("lxml==4.9.2")
            from lxml import etree

            parser = etree.HTMLParser()
            for credit, show_on_screen in asset_credits:
                if credit == "" or show_on_screen is not should_show_on_screen:
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

            if perform_fallback:
                self._logger.warning("Performing credits fallback.")
                for credit, _ in self._credits:
                    ui.Label(credit, height=0, word_wrap=True)
