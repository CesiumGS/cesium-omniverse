import omni.ui as ui

READ_ONLY_STYLE = {"color": ui.color("#888888")}


def string_field_with_label(label_text, model=None, enabled=True):
    with ui.HStack(spacing=4, height=20):
        ui.Label(label_text, height=20, width=100)
        field = ui.StringField(height=20, enabled=enabled)
        if not enabled:
            field.style = READ_ONLY_STYLE
        if model:
            field.model = model
        return field


def int_field_with_label(label_text, model=None, enabled=True):
    with ui.HStack(spacing=4, height=20):
        ui.Label(label_text, height=20, width=100)
        field = ui.IntField(height=20, enabled=enabled)
        if not enabled:
            field.style = READ_ONLY_STYLE
        if model:
            field.model = model
        return field


def float_field_with_label(label_text, model=None, enabled=True):
    with ui.HStack(spacing=4, height=20):
        ui.Label(label_text, height=20, width=100)
        field = ui.FloatField(height=20, enabled=enabled, precision=7)
        if not enabled:
            field.style = READ_ONLY_STYLE
        if model:
            field.model = model
        return field
