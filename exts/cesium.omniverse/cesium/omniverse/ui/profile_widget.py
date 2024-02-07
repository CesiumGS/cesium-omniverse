import omni.usd
import logging
import carb.events
import omni.kit.app as app
import omni.ui as ui
from typing import List, Optional
from ..bindings import ICesiumOmniverseInterface, CesiumIonSession
from enum import Enum
from cesium.usd.plugins.CesiumUsdSchemas import IonServer as CesiumIonServer
from ..usdUtils import set_path_to_current_ion_server


class SessionState(Enum):
    NOT_CONNECTED = 1
    LOADING = 2
    CONNECTED = 3


def get_session_state(session: CesiumIonSession) -> SessionState:
    if session.is_profile_loaded():
        return SessionState.CONNECTED
    elif session.is_loading_profile():
        return SessionState.LOADING
    else:
        return SessionState.NOT_CONNECTED


def get_profile_id(session: CesiumIonSession) -> Optional[int]:
    if session.is_profile_loaded():
        profile = session.get_profile()
        return profile.id

    return None


class SessionComboItem(ui.AbstractItem):
    def __init__(self, session: CesiumIonSession, server: CesiumIonServer):
        super().__init__()

        session_state = get_session_state(session)
        prefix = ""
        suffix = ""

        if session_state == SessionState.NOT_CONNECTED:
            suffix += " (not connected)"
        elif session_state == SessionState.LOADING:
            suffix += " (loading profile...)"
        elif session_state == SessionState.CONNECTED:
            prefix += session.get_profile().username
            prefix += " @ "

        # Get the display name from the server prim. If that's empty, use the prim path.
        server_name = server.GetDisplayNameAttr().Get()
        if server_name == "":
            server_name = server.GetPath()

        self.text = ui.SimpleStringModel(f"{prefix}{server_name}{suffix}")
        self.server = server


class SessionComboModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)

        self._current_index = ui.SimpleIntModel(0)
        self._current_index.add_value_changed_fn(lambda index_model: self._item_changed(None))

        self._items = []

    def replace_all_items(
        self, sessions: List[CesiumIonSession], servers: List[CesiumIonServer], current_server: CesiumIonServer
    ):
        self._items.clear()
        self._items = [SessionComboItem(session, server) for session, server in zip(sessions, servers)]

        current_index = 0
        for index, server in enumerate(servers):
            if server.GetPath() == current_server.GetPath():
                current_index = index
                break

        self._current_index.set_value(current_index)
        self._item_changed(None)

    def get_item_children(self, item=None):
        return self._items

    def get_item_value_model(self, item: SessionComboItem = None, column_id: int = 0):
        if item is None:
            return self._current_index
        return item.text

    def get_current_selection(self):
        if len(self._items) < 1:
            return None

        return self._items[self._current_index.get_value_as_int()]


class CesiumOmniverseProfileWidget(ui.Frame):
    def __init__(self, cesium_omniverse_interface: ICesiumOmniverseInterface, **kwargs):
        self._logger = logging.getLogger(__name__)
        self._cesium_omniverse_interface = cesium_omniverse_interface

        self._profile_ids: List[int] = []
        self._session_states: List[SessionState] = []
        self._server_paths: List[str] = []
        self._server_names: List[str] = []

        self._sessions_combo_box: Optional[ui.ComboBox] = None
        self._sessions_combo_model = SessionComboModel()
        self._sessions_combo_model.add_item_changed_fn(self._on_item_changed)

        self._subscriptions: List[carb.events.ISubscription] = []
        self._setup_subscriptions()

        super().__init__(build_fn=self._build_ui, **kwargs)

    def destroy(self) -> None:
        for subscription in self._subscriptions:
            subscription.unsubscribe()
        self._subscriptions.clear()

    def _setup_subscriptions(self):
        update_stream = app.get_app().get_update_event_stream()
        self._subscriptions.append(
            update_stream.create_subscription_to_pop(self._on_update_frame, name="on_update_frame")
        )

    def _on_item_changed(self, item_model, item):
        item = self._sessions_combo_model.get_current_selection()
        if item is not None:
            server_path = item.server.GetPath()
            set_path_to_current_ion_server(server_path)

    def _on_update_frame(self, _e: carb.events.IEvent):
        if omni.usd.get_context().get_stage_state() != omni.usd.StageState.OPENED:
            return

        stage = omni.usd.get_context().get_stage()

        sessions = self._cesium_omniverse_interface.get_sessions()
        server_paths = self._cesium_omniverse_interface.get_server_paths()
        servers = [CesiumIonServer.Get(stage, server_path) for server_path in server_paths]
        server_names = [server.GetDisplayNameAttr().Get() for server in servers]
        current_server_path = self._cesium_omniverse_interface.get_server_path()
        current_server = CesiumIonServer.Get(stage, current_server_path)

        profile_ids = []
        session_states = []

        for session in sessions:
            profile_id = get_profile_id(session)
            session_state = get_session_state(session)

            if session.is_connected() and not session.is_profile_loaded():
                session.refresh_profile()

            profile_ids.append(profile_id)
            session_states.append(session_state)

        if (
            profile_ids != self._profile_ids
            or session_states != self._session_states
            or server_paths != self._server_paths
            or server_names != self._server_names
        ):
            self._logger.info("Rebuilding profile widget")

            self._profile_ids = profile_ids
            self._session_states = session_states
            self._server_paths = server_paths
            self._server_names = server_names

            self._sessions_combo_model.replace_all_items(sessions, servers, current_server)

            self.rebuild()

    def _build_ui(self):
        self._sessions_combo_box = ui.ComboBox(self._sessions_combo_model)
