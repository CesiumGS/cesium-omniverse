import logging
import omni.kit.pipapi
import omni.ui as ui
import webbrowser
from dataclasses import dataclass
from functools import partial
from typing import Optional, List, Tuple
from .uri_image import CesiumUriImage
from .image_button import CesiumImageButton

_num_retries = 0


@dataclass
class ParsedCredit:
    text: Optional[str] = None
    image_uri: Optional[str] = None
    link: Optional[str] = None


class CesiumCreditsParser:
    """Takes in a credits array and outputs the elements necessary to show the credits.
    Should be embedded in a VStack or HStack."""

    def _parse_element(self, element, link: Optional[str] = None) -> List[ParsedCredit]:
        results = []

        tag = element.tag
        if tag == "html" or tag == "body":
            for child in element.iterchildren():
                results.extend(self._parse_element(child, link))
        elif tag == "a":
            # TODO: We probably need to do some sanitization of the href.
            link = element.attrib["href"]
            text = "".join(element.itertext())

            if text != "":
                results.append(ParsedCredit(text=text, link=link))
            for child in element.iterchildren():
                results.extend(self._parse_element(child, link))
        elif tag == "img":
            src = element.attrib["src"]
            if link is None:
                results.append(ParsedCredit(image_uri=src))
            else:
                results.append(ParsedCredit(image_uri=src, link=link))
        elif tag == "span" or tag == "div":
            for child in element.iterchildren():
                results.extend(self._parse_element(child, link))

            # Sometimes divs or spans have text.
            text = "".join(element.itertext())
            if text:
                results.append(ParsedCredit(text=text))
        else:
            text = "".join(element.itertext())
            if link is None:
                results.append(ParsedCredit(text=text))
            else:
                results.append(ParsedCredit(text=text, link=link))

        return results

    def _parse_credits(
        self, asset_credits: List[Tuple[str, bool]], should_show_on_screen: bool, perform_fallback=False
    ) -> List[ParsedCredit]:
        results = []

        global _num_retries
        if _num_retries > 10:
            # Once we've attempted 10 times, we don't want to attempt again for the life of this session.
            return results

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
                        results.extend(self._parse_element(doc))
                        continue
                    except etree.XMLSyntaxError as err:
                        self._logger.info(err)

                results.append(ParsedCredit(text=credit))

        except Exception as e:
            self._logger.debug(e)

            _num_retries = _num_retries + 1

            if perform_fallback:
                self._logger.warning("Performing credits fallback.")
                for credit, _ in asset_credits:
                    results.append(ParsedCredit(text=credit))

        return results

    @staticmethod
    def _button_clicked(link: str):
        webbrowser.open_new_tab(link)

    def _build_ui_elements(self, parsed_credits: List[ParsedCredit], label_alignment: ui.Alignment):
        for parsed_credit in parsed_credits:
            if parsed_credit.image_uri is not None:
                if parsed_credit.link is not None:
                    CesiumImageButton(
                        src=parsed_credit.image_uri,
                        padding=4,
                        clicked_fn=partial(self._button_clicked, parsed_credit.link),
                    )
                else:
                    CesiumUriImage(src=parsed_credit.image_uri)
            elif parsed_credit.text is not None:
                if parsed_credit.link is not None:
                    ui.Button(
                        parsed_credit.text,
                        clicked_fn=partial(self._button_clicked, parsed_credit.link),
                        height=0,
                        width=0,
                    )
                else:
                    ui.Label(parsed_credit.text, height=0, word_wrap=True, alignment=label_alignment)

    def _build_ui(self, parsed_credits: List[ParsedCredit], combine_labels: bool, label_alignment: ui.Alignment):
        if combine_labels:
            label_strings = []
            other_credits = []

            for credit in parsed_credits:
                if credit.text is not None and credit.link is None:
                    label_strings.append(credit.text)
                else:
                    other_credits.append(credit)

            label_strings_combined = " - ".join(label_strings)

            # Add the label even if the string is empty. The label will expand to fill the parent HStack
            # which acts like a spacer that right-aligns the image and button elements. Eventually we
            # should find a different solution here.
            ui.Label(label_strings_combined, height=0, word_wrap=True, alignment=label_alignment)

            self._build_ui_elements(other_credits, label_alignment)
        else:
            self._build_ui_elements(parsed_credits, label_alignment)

    # There is a builtin name called credits, which is why this argument is called asset_credits.
    def __init__(
        self,
        asset_credits: List[Tuple[str, bool]],
        should_show_on_screen: bool,
        perform_fallback=False,
        combine_labels=False,
        label_alignment=ui.Alignment.LEFT,
    ):
        self._logger = logging.getLogger(__name__)

        parsed_credits = self._parse_credits(asset_credits, should_show_on_screen, perform_fallback)
        self._build_ui(parsed_credits, combine_labels, label_alignment)
