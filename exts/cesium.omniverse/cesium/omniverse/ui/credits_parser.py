import logging
import omni.kit.pipapi
import omni.ui as ui
import omni.kit.viewport.utility as viewport_util
import webbrowser
from typing import Optional, List, Tuple
from .uri_image import CesiumUriImage
from .image_button import CesiumImageButton

_num_retries = 0


class CesiumCreditsParser:
    """Takes in a credits array and outputs the elements necessary to show the credits.
    Should be embedded in a VStack or HStack."""

    @staticmethod
    def _calc_label_width(wrap_early):
        resolution_width = viewport_util.get_active_viewport().resolution[0]
        return resolution_width * 0.40 if wrap_early else 0

    def _parse_element(self, element, link: Optional[str] = None, wrap_early=False, in_window=False):
        # Parameters wrap_early and in_window are hacks for making everything layout correctly. The wrap_early
        # parameter is used to wrap elements early for the "all credits onscreen" functionality, and the in_window
        # parameter is to fix an edge case bug that occurs in the "Data Attribution" window because width=None and
        # width=0 are not equal in Omniverse UI land when dealing with label word wrapping.
        tag = element.tag
        label_width = self._calc_label_width(wrap_early)
        if tag == "html" or tag == "body":
            for child in element.iterchildren():
                self._parse_element(child, link, wrap_early, in_window)
        elif tag == "a":
            # TODO: We probably need to do some sanitization of the href.
            link = element.attrib["href"]
            text = "".join(element.itertext())

            if text != "":
                ui.Button(text, height=0, width=0, clicked_fn=lambda: webbrowser.open_new_tab(link))
            for child in element.iterchildren():
                self._parse_element(child, link, wrap_early, in_window)
        elif tag == "img":
            src = element.attrib["src"]
            if link is None:
                CesiumUriImage(src=src)
            else:
                CesiumImageButton(src=src, padding=4, clicked_fn=lambda: webbrowser.open_new_tab(link))
        elif tag == "span" or tag == "div":
            for child in element.iterchildren():
                self._parse_element(child, link, wrap_early, in_window)

            # Sometimes divs or spans have text.
            text = "".join(element.itertext())
            if text:
                if in_window:
                    ui.Label(text, height=0, word_wrap=True)
                else:
                    ui.Label(text, height=0, width=label_width, word_wrap=True)
        else:
            text = "".join(element.itertext())
            if link is None:
                if in_window:
                    ui.Label(text, height=0, word_wrap=True)
                else:
                    ui.Label(text, height=0, width=label_width, word_wrap=True)
            else:
                ui.Button(text, clicked_fn=lambda: webbrowser.open_new_tab(link), height=0, width=0)

    # There is a builtin name called credits, which is why this argument is called asset_credits.
    def __init__(
        self,
        asset_credits: List[Tuple[str, bool]],
        should_show_on_screen: bool,
        perform_fallback=False,
        wrap_early=False,
        in_window=False,
    ):
        self._logger = logging.getLogger(__name__)

        global _num_retries
        if _num_retries > 10:
            # Once we've attempted 10 times, we don't want to attempt again for the life of this session.
            return

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
                        self._parse_element(doc, wrap_early=wrap_early, in_window=in_window)
                        continue
                    except etree.XMLSyntaxError as err:
                        self._logger.info(err)

                ui.Label(credit, height=0, word_wrap=True)

        except Exception as e:
            self._logger.debug(e)

            _num_retries = _num_retries + 1

            if perform_fallback:
                self._logger.warning("Performing credits fallback.")
                for credit, _ in asset_credits:
                    ui.Label(credit, height=0, word_wrap=True)
