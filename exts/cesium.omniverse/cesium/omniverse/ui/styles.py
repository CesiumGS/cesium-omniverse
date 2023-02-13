from omni.ui import Alignment, color as cl, Direction


class CesiumOmniverseUiStyles:
    intro_label_style = {
        "font_size": 16,
    }

    quick_add_section_label = {
        "font_size": 20,
        "margin": 5,
    }

    quick_add_button = {
        "Button.Label": {
            "font_size": 16
        }
    }

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
        "Button:pressed": {
            "background_color": cl("#2D6179")
        }
    }

    top_bar_button_style = {
        "Button": {
            "padding": 10.0,
            "stack_direction": Direction.TOP_TO_BOTTOM
        },
        "Button.Image": {
            "alignment": Alignment.CENTER,
        },
        "Button.Label": {
            "alignment": Alignment.CENTER_BOTTOM
        }
    }
