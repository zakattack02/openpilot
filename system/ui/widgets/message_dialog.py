import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.button import gui_button, ButtonStyle
from openpilot.system.ui.lib.label import gui_text_box

# Constants for dialog dimensions and styling
DIALOG_WIDTH = 1520
DIALOG_HEIGHT = 600
BUTTON_HEIGHT = 160
BUTTON_WIDTH = 320
MARGIN = 50
TEXT_AREA_HEIGHT_REDUCTION = 200
BACKGROUND_COLOR = rl.Color(27, 27, 27, 255)


def message_dialog(message: str, button_text: str = "OK") -> int:
    dialog_x = (gui_app.width - DIALOG_WIDTH) / 2
    dialog_y = (gui_app.height - DIALOG_HEIGHT) / 2
    dialog_rect = rl.Rectangle(dialog_x, dialog_y, DIALOG_WIDTH, DIALOG_HEIGHT)

    # Calculate button position (centered at the bottom of the dialog)
    bottom = dialog_rect.y + dialog_rect.height
    button_x = dialog_rect.x + (dialog_rect.width - BUTTON_WIDTH) / 2
    button_y = bottom - BUTTON_HEIGHT - MARGIN
    button_rect = rl.Rectangle(button_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT)

    # Draw the dialog background
    rl.draw_rectangle_rec(dialog_rect, BACKGROUND_COLOR)

    # Draw the message in the dialog, centered
    text_rect = rl.Rectangle(dialog_rect.x, dialog_rect.y, dialog_rect.width, dialog_rect.height - TEXT_AREA_HEIGHT_REDUCTION)
    gui_text_box(
        text_rect,
        message,
        font_size=88,
        alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER,
        alignment_vertical=rl.GuiTextAlignmentVertical.TEXT_ALIGN_MIDDLE,
    )

    # Initialize result; -1 means no action taken yet
    result = -1

    # Check for keyboard input for accessibility
    if rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
      result = 1
    elif rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE):
      result = 1

    # Check for button click
    if gui_button(button_rect, button_text, button_style=ButtonStyle.PRIMARY):
      result = 1

    return result
