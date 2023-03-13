# -----------------------------
# Options effecting formatting.
# -----------------------------
with section("format"):

    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If a positional argument group contains more than this many arguments, then
    # force it to a vertical layout.
    max_pargs_hwrap = 3

with section("markup"):

    # Disable formatting entirely, making cmake-format a no-op
    enable_markup = False
