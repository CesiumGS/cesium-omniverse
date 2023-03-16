import logging
import carb.events
import omni.kit.app as app
import omni.kit.pipapi
import omni.ui as ui
from omni.kit.viewport.utility import get_active_viewport_window
import webbrowser
from typing import List, Optional
from .uri_image import CesiumUriImage
from ..bindings import ICesiumOmniverseInterface
from .credits_window import CesiumOmniverseCreditsWindow
from .image_button import CesiumImageButton


class CesiumCreditsViewportFrame:
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface):
        self._logger = logging.getLogger(__name__)

        self._cesium_omniverse_interface = cesium_omniverse_interface

        viewport_window = get_active_viewport_window()
        self._credits_viewport_frame = viewport_window.get_frame("cesium.omniverse.viewport.ION_CREDITS")

        self._credits_window: Optional[CesiumOmniverseCreditsWindow] = None
        self._data_attribution_button: Optional[ui.Button] = None

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        self._credits: List[(str, bool)] = []

        self._build_fn()

    def __del__(self):
        self.destroy()

    def destroy(self):
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

        if self._credits_window is not None:
            self._credits_window.destroy()
            self._credits_window = None

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(
                self._on_update_frame, name="cesium.omniverse.viewport.ON_UPDATE_FRAME"
            )
        )

    def _on_update_frame(self, _e: carb.events.IEvent):
        if self._data_attribution_button is None:
            return

        credits_available = self._cesium_omniverse_interface.credits_available()

        if credits_available != self._data_attribution_button.visible:
            if credits_available:
                self._logger.info("Show Data Attribution")
            else:
                self._logger.info("Hide Data Attribution")
            self._data_attribution_button.visible = credits_available

        if self._data_attribution_button.visible:
            new_credits = self._cesium_omniverse_interface.get_credits()
            if new_credits is not None and len(self._credits) != len(new_credits):
                self._credits.clear()
                self._credits.extend(new_credits)
                self._build_fn()
        else:
            self._credits.clear()
            self._build_fn()

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
                CesiumImageButton(src=src, clicked_fn=lambda: webbrowser.open(link))
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

    def _on_data_attribution_button_clicked(self):
        self._credits_window = CesiumOmniverseCreditsWindow(self._cesium_omniverse_interface)

    def _build_fn(self):
        with self._credits_viewport_frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=0):
                    ui.Spacer()

                    try:
                        omni.kit.pipapi.install("lxml==4.9.2")
                        from lxml import etree

                        parser = etree.HTMLParser()
                        for credit, showOnScreen in self._credits:
                            if credit == "" or showOnScreen is False:
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

                    self._data_attribution_button = ui.Button(
                        "Data Attribution",
                        visible=False,
                        width=0,
                        height=0,
                        clicked_fn=self._on_data_attribution_button_clicked,
                    )
