from omni.ui import Alignment, color as cl, Direction


class CesiumOmniverseUiStyles:
    intro_label_style = {
        "font_size": 16,
    }

    troubleshooter_header_style = {
        "font_size": 18,
    }

    attribution_header_style = {
        "font_size": 18,
    }

    quick_add_section_label = {
        "font_size": 20,
        "margin": 5,
    }

    quick_add_button = {"Button.Label": {"font_size": 16}}

    blue_button_style = {
        "Button": {
            "background_color": cl("#4BA1CA"),
            "padding": 12,
        },
        "Button.Label": {
            "color": cl("#FFF"),
            "font_size": 16,
        },
        "Button:hovered": {
            "background_color": cl("#3C81A2"),
        },
        "Button:pressed": {"background_color": cl("#2D6179")},
    }

    top_bar_button_style = {
        "Button": {"padding": 10.0, "stack_direction": Direction.TOP_TO_BOTTOM},
        "Button.Image": {
            "alignment": Alignment.CENTER,
        },
        "Button.Label": {"alignment": Alignment.CENTER_BOTTOM},
        "Button.Image:disabled": {"color": cl("#808080")},
        "Button.Label:disabled": {"color": cl("#808080")},
    }

    asset_detail_frame = {"ScrollingFrame": {"background_color": cl("#1F2123"), "padding": 10}}

    asset_detail_name_label = {"font_size": 22}

    asset_detail_header_label = {"font_size": 18}

    asset_detail_id_label = {"font_size": 14}

    asset_detail_content_label = {"font_size": 16}
