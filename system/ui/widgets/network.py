from dataclasses import dataclass
from typing import Literal
import time

import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.button import ButtonStyle, gui_button
from openpilot.system.ui.lib.label import gui_label
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.wifi_manager import NetworkInfo, WifiManagerCallbacks, WifiManagerWrapper, SecurityType
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.confirm_dialog import confirm_dialog
from openpilot.system.ui.widgets.message_dialog import message_dialog

NM_DEVICE_STATE_NEED_AUTH = 60
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 64
ITEM_HEIGHT = 160
ICON_SIZE = 49

CONNECTION_TIMEOUT = 30.0
FORGET_TIMEOUT = 10.0

STRENGTH_ICONS = [
  "icons/wifi_strength_low.png",
  "icons/wifi_strength_medium.png",
  "icons/wifi_strength_high.png",
  "icons/wifi_strength_full.png",
]

@dataclass
class StateIdle:
  action: Literal["idle"] = "idle"

@dataclass
class StateConnecting:
  network: NetworkInfo
  action: Literal["connecting"] = "connecting"
  start_time: float = 0.0  # Track when connection attempt started
@dataclass
class StateConnectionError:
  network: NetworkInfo
  message: str
  action: Literal["connection_error"] = "connection_error"

@dataclass
class StateNeedsAuth:
  network: NetworkInfo
  action: Literal["needs_auth"] = "needs_auth"

@dataclass
class StateShowForgetConfirm:
  network: NetworkInfo
  action: Literal["show_forget_confirm"] = "show_forget_confirm"

@dataclass
class StateForgetting:
  network: NetworkInfo
  action: Literal["forgetting"] = "forgetting"
  start_time: float = 0.0  # Track when forget attempt started

UIState = StateIdle | StateConnecting | StateConnectionError | StateNeedsAuth | StateShowForgetConfirm | StateForgetting


class WifiManagerUI:
  def __init__(self, wifi_manager: WifiManagerWrapper):
    self.state: UIState = StateIdle()
    self.btn_width = 200
    self.scroll_panel = GuiScrollPanel()
    self.keyboard = Keyboard(max_text_size=MAX_PASSWORD_LENGTH, min_text_size=MIN_PASSWORD_LENGTH, show_password_toggle=True)

    self._networks: list[NetworkInfo] = []
    self._last_refresh_time = 0.0
    self._pending_network_update = False

    self.wifi_manager = wifi_manager
    self.wifi_manager.set_callbacks(WifiManagerCallbacks(
      self._on_need_auth,
      self._on_activated,
      self._on_forgotten,
      self._on_network_updated
    ))
    self.wifi_manager.start()
    self.wifi_manager.connect()

  def render(self, rect: rl.Rectangle):
    # Check for timeouts on stateful operations
    self._check_timeouts()

    if not self._networks:
      gui_label(rect, "Scanning Wi-Fi networks...", 72, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
      return

    match self.state:
      case StateNeedsAuth(network):
        result = self.keyboard.render("Enter password", f"for {network.ssid}")
        if result == 1:
          password = self.keyboard.text
          self.keyboard.clear()

          if len(password) >= MIN_PASSWORD_LENGTH:
            self.connect_to_network(network, password)
        elif result == 0:
          self.state = StateIdle()

      case StateConnectionError(_, message):
        result = message_dialog(f"Connection Error: {message}")
        if result == 1:
          self.state = StateIdle()

      case StateShowForgetConfirm(network):
        result = confirm_dialog(f'Forget Wi-Fi Network "{network.ssid}"?', "Forget")
        if result == 1:
          self.forget_network(network)
        elif result == 0:
          self.state = StateIdle()

      case _:
        self._draw_network_list(rect)
  def _check_timeouts(self):
    """Check for timeouts on operations and transition states accordingly"""
    current_time = time.time()

    match self.state:
      case StateConnecting(network, start_time=start_time):
        if current_time - start_time > CONNECTION_TIMEOUT:
          self.state = StateConnectionError(network, "Connection attempt timed out")

      case StateForgetting(network, start_time=start_time):
        if current_time - start_time > FORGET_TIMEOUT:
          self.state = StateConnectionError(network, "Failed to forget network")

    # Refresh network list periodically if needed
    if self._pending_network_update and (current_time - self._last_refresh_time > 2.0):
      self._pending_network_update = False
      self._last_refresh_time = current_time
      self.wifi_manager.request_scan()

  @property
  def require_full_screen(self) -> bool:
    """Check if the WiFi UI requires exclusive full-screen rendering."""
    return isinstance(self.state, (StateNeedsAuth, StateShowForgetConfirm, StateConnectionError))

  def _draw_network_list(self, rect: rl.Rectangle):
    content_rect = rl.Rectangle(rect.x, rect.y, rect.width, len(self._networks) * ITEM_HEIGHT)
    offset = self.scroll_panel.handle_scroll(rect, content_rect)
    clicked = self.scroll_panel.is_click_valid()

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    for i, network in enumerate(self._networks):
      y_offset = rect.y + i * ITEM_HEIGHT + offset.y
      item_rect = rl.Rectangle(rect.x, y_offset, rect.width, ITEM_HEIGHT)
      if not rl.check_collision_recs(item_rect, rect):
        continue

      self._draw_network_item(item_rect, network, clicked)
      if i < len(self._networks) - 1:
        line_y = int(item_rect.y + item_rect.height - 1)
        rl.draw_line(int(item_rect.x), int(line_y), int(item_rect.x + item_rect.width), line_y, rl.LIGHTGRAY)

    rl.end_scissor_mode()

  def _draw_network_item(self, rect, network: NetworkInfo, clicked: bool):
    spacing = 50
    ssid_rect = rl.Rectangle(rect.x, rect.y, rect.width - self.btn_width * 2, ITEM_HEIGHT)
    signal_icon_rect = rl.Rectangle(rect.x + rect.width - ICON_SIZE, rect.y + (ITEM_HEIGHT - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)
    security_icon_rect = rl.Rectangle(signal_icon_rect.x - spacing - ICON_SIZE, rect.y + (ITEM_HEIGHT - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)

    gui_label(ssid_rect, network.ssid, 55)

    status_text = ""
    match self.state:
      case StateConnecting(network=connecting):
        if connecting.ssid == network.ssid:
          status_text = "CONNECTING..."
      case StateForgetting(network=forgetting):
        if forgetting.ssid == network.ssid:
          status_text = "FORGETTING..."

    if status_text:
      status_text_rect = rl.Rectangle(security_icon_rect.x - 410, rect.y, 410, ITEM_HEIGHT)
      gui_label(status_text_rect, status_text, font_size=48, alignment=rl.GuiTextAlignment.TEXT_ALIGN_CENTER)
    elif network.is_saved:
      # If the network is saved, show the "Forget" button
      forget_btn_rect = rl.Rectangle(security_icon_rect.x - self.btn_width - spacing,
        rect.y + (ITEM_HEIGHT - 80) / 2,
        self.btn_width,
        80,
      )
      if gui_button(forget_btn_rect, "Forget", button_style=ButtonStyle.ACTION, is_enabled=isinstance(self.state, StateIdle)) and clicked:
        self.state = StateShowForgetConfirm(network)

    # Always draw status and signal strength icons
    self._draw_status_icon(security_icon_rect, network)
    self._draw_signal_strength_icon(signal_icon_rect, network)

    # Handle clicking on the network SSID - only in idle state
    if isinstance(self.state, StateIdle) and rl.check_collision_point_rec(rl.get_mouse_position(), ssid_rect) and clicked:
      if network.security_type == SecurityType.UNSUPPORTED:
        # Show error message for unsupported security types (like WPA3)
        self.state = StateConnectionError(network, "Unsupported security type")
      elif not network.is_saved and network.security_type != SecurityType.OPEN:
        # Need password for non-saved secured networks
        self.state = StateNeedsAuth(network)
      elif not network.is_connected:
        # Connect to saved networks or open networks directly
        self.connect_to_network(network)

  def _draw_status_icon(self, rect, network: NetworkInfo):
    """Draw the status icon based on network's connection state"""
    icon_file = None
    if network.is_connected:
      icon_file = "icons/checkmark.png"
    elif network.security_type == SecurityType.UNSUPPORTED:
      icon_file = "icons/circled_slash.png"
    elif network.security_type != SecurityType.OPEN:
      icon_file = "icons/lock_closed.png"

    if not icon_file:
      return

    texture = gui_app.texture(icon_file, ICON_SIZE, ICON_SIZE)
    icon_rect = rl.Vector2(rect.x, rect.y + (ICON_SIZE - texture.height) / 2)
    rl.draw_texture_v(texture, icon_rect, rl.WHITE)

  def _draw_signal_strength_icon(self, rect: rl.Rectangle, network: NetworkInfo):
    """Draw the Wi-Fi signal strength icon based on network's signal strength"""
    strength_level = max(0, min(3, round(network.strength / 33.0)))
    rl.draw_texture_v(gui_app.texture(STRENGTH_ICONS[strength_level], ICON_SIZE, ICON_SIZE), rl.Vector2(rect.x, rect.y), rl.WHITE)

  def connect_to_network(self, network: NetworkInfo, password=''):
    # Start connection attempt with timeout tracking
    self.state = StateConnecting(network=network, start_time=time.time())

    if network.is_saved and not password:
      self.wifi_manager.activate_connection(network.ssid)
    else:
      self.wifi_manager.connect_to_network(network.ssid, password)

    # Request network scan after a short delay to ensure UI state is updated
    self._pending_network_update = True
    self._last_refresh_time = time.time()

  def forget_network(self, network: NetworkInfo):
    # Start forget operation with timeout tracking
    self.state = StateForgetting(network=network, start_time=time.time())

    # Don't modify network.is_saved locally, wait for backend to confirm
    self.wifi_manager.forget_connection(network.ssid)

    # Request network scan after a short delay to update UI state
    self._pending_network_update = True
    self._last_refresh_time = time.time()

  def _on_network_updated(self, networks: list[NetworkInfo]):
    self._networks = networks
    self._last_refresh_time = time.time()

    # If we're in certain states, update our state to reflect network changes
    match self.state:
      case StateConnecting(network):
        # Check if the network is now connected
        updated_network = next((n for n in networks if n.ssid == network.ssid), None)
        if updated_network and updated_network.is_connected:
          self.state = StateIdle()

  def _on_need_auth(self, ssid):
    # When a network needs authentication
    # First get the full network info by ssid
    network = next((n for n in self._networks if n.ssid == ssid), None)
    if not network:
      return
    match self.state:
      case StateConnecting(connecting_network) if connecting_network.ssid == ssid:
        # We were trying to connect to this network, now need auth
        self.state = StateNeedsAuth(network)
      case _:
        # Otherwise, transition to auth state for this network
        self.state = StateNeedsAuth(network)

  def _on_activated(self):
    # A connection was successfully activated
    match self.state:
      case StateConnecting(_):
        # Successfully connected - move to idle state
        self.state = StateIdle()
        # Force refresh networks to show updated connection status
        self._pending_network_update = True
        self._last_refresh_time = 0.0  # Ensure immediate refresh
      case _:
        # Even if we weren't explicitly connecting, refresh networks
        self._pending_network_update = True

  def _on_forgotten(self):
    # A network was successfully forgotten
    match self.state:
      case StateForgetting(_):
        # Successfully forgot network - move to idle state
        self.state = StateIdle()
        # Force refresh networks list to show updated saved status
        self._pending_network_update = True
        self._last_refresh_time = 0.0  # Ensure immediate refresh


def main():
  gui_app.init_window("Wi-Fi Manager")
  wifi_manager = WifiManagerWrapper()
  wifi_ui = WifiManagerUI(wifi_manager)

  for _ in gui_app.render():
    wifi_ui.render(rl.Rectangle(50, 50, gui_app.width - 100, gui_app.height - 100))

  wifi_manager.shutdown()
  gui_app.close()


if __name__ == "__main__":
  main()
