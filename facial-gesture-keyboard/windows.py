"""
Native window orchestration for the standalone desktop app.

Three pywebview windows:
  - main:    the calibration/launch page (camera, calibration, Confirm
             button). A normal window - it's fine for this one to take
             focus when the user interacts with it directly.
  - keyboard: the on-screen keyboard, an always-on-top overlay that
             must NOT take OS keyboard focus when shown, or every
             gesture-typed keystroke would land on this window instead
             of whatever app (Word, a browser, ...) the user is
             actually typing into.
  - toggle:  a small tab docked to the left screen edge, shown once the
             user has confirmed calibration; clicking it reopens the
             main window from either other mode.

Three modes (DesktopWindows.mode), matching the actual product design:
  - "setup"    - the main/calibration window is visible; nothing else
                 is. Entered at launch, or via the toggle tab from
                 either other mode.
  - "eye"      - eye/gaze cursor mode (gaze-to-cursor tracking itself
                 is a later pass; this just wires the mode and its
                 gestures). Receives left_click/right_click (real OS
                 mouse clicks) and switch_to_keyboard (brings up the
                 keyboard - the only gesture eye mode listens for
                 besides the two clicks). Entered from "setup" via the
                 Confirm button - this, not the keyboard, is the
                 resting state after confirming: you land able to
                 control the cursor, and bring up the keyboard on
                 demand, rather than the other way around.
  - "keyboard" - the keyboard overlay is visible and receives
                 up/down/left/right/confirm/backspace gestures. Entered
                 from "eye" via the switch_to_keyboard gesture. The
                 *only* way back out to eye mode is the on-screen
                 "toggle" key inside the keyboard grid itself (a normal
                 navigate+confirm key press, not a gesture) - there is
                 deliberately no gesture that hides the keyboard from
                 within keyboard mode.

Keyboard-mode gestures and eye-mode gestures are never listened to
simultaneously - on_gesture() below gates on self.mode so a gesture
fired while in the "wrong" mode is simply ignored, exactly as spec'd
("we ignore the one that we aren't currently using"). This is enforced
a layer earlier too: every mode change calls into main.py
(server_module.set_active_mode) so the live classifier itself only ever
scores the labels reachable from the current mode - see
CalibrationStore.predict in gestures.py for why that's not just
belt-and-suspenders (it changes what the model actually competes over,
not just what happens with the result).

The core trick this all depends on - showing a window without stealing
focus - was verified empirically before building this: pywebview's
`.show()` always activates the window even with the WS_EX_NOACTIVATE
style set (that style only stops *clicks* from activating it, not
Show() itself), so the working approach is "let it activate, then
immediately hand focus back" to whatever had it a moment before. In
testing this restore happens within ~150ms and holds indefinitely
afterward (checked continuously for 5s with an actively-rendering page
in the window) - the previously-focused app never visibly loses focus
in any way a user watching for it could notice while typing.

Also depends on: `window.hide()` does NOT throttle the page's JS timers
the way a backgrounded browser TAB does (verified separately: a
setInterval(..., 50) ticked at the full, unthrottled rate across 5s
while hidden) - so the main window's camera/gesture-recognition loop
keeps running normally after Confirm hides it.
"""

from __future__ import annotations

import ctypes
import time

import webview
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

import main as server_module

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010


def _apply_noactivate_topmost(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


def _get_foreground_hwnd() -> int:
    return ctypes.windll.user32.GetForegroundWindow()


def show_without_stealing_focus(window: webview.Window) -> None:
    """Show an overlay window without taking OS keyboard focus away
    from whatever the user was previously working in."""
    previously_focused = _get_foreground_hwnd()
    hwnd = window.native.Handle.ToInt64()
    _apply_noactivate_topmost(hwnd)
    window.show()
    # Show() always activates the window regardless of WS_EX_NOACTIVATE -
    # hand focus back immediately rather than trying to prevent the
    # (very brief) activation in the first place.
    time.sleep(0.05)
    ctypes.windll.user32.SetForegroundWindow(previously_focused)


KEYBOARD_MODE_GESTURES = {"up", "down", "left", "right", "confirm", "backspace"}


class DesktopWindows:
    """Owns the three windows, the current mode, and the transitions
    between them. Mouse control (eye mode's clicks) lives here too,
    alongside the mode logic, rather than in main.py - it's purely
    mode-orchestration, not gesture recognition."""

    def __init__(self, main_window: webview.Window, keyboard_window: webview.Window, toggle_window: webview.Window) -> None:
        self.main_window = main_window
        self.keyboard_window = keyboard_window
        self.toggle_window = toggle_window
        self.mode = "setup"
        self.mouse = MouseController()
        server_module.set_active_mode(self.mode)

    def _set_mode(self, mode: str) -> None:
        """Every mode transition goes through here so the server's live
        classifier (see main.py's active_mode / MODE_ALLOWED_LABELS)
        never falls out of sync with what's actually on screen."""
        self.mode = mode
        server_module.set_active_mode(mode)

    # --- called from the main window's js_api (Confirm button + gestures) --

    def confirm_calibration(self) -> None:
        """Confirm button on the launch page: minimizes this window to
        the left-edge tab and drops straight into eye/cursor mode - not
        the keyboard. Starting in keyboard mode with no way to move a
        cursor isn't a realistic resting state; landing in cursor
        control and bringing up the keyboard on demand (via the
        switch_to_keyboard gesture) is."""
        self.main_window.hide()
        show_without_stealing_focus(self.toggle_window)
        self._set_mode("eye")

    def on_gesture(self, label: str) -> None:
        """
        Every fired gesture from the main window's recognition loop
        arrives here, regardless of which window is currently visible -
        confirmed separately that hide() doesn't pause the main
        window's JS. Only the gestures belonging to the *current* mode
        are acted on; everything else is ignored.
        """
        if self.mode == "keyboard":
            if label in KEYBOARD_MODE_GESTURES:
                safe_label = label.replace("\\", "").replace("'", "")
                self.keyboard_window.evaluate_js(f"applyGestureToKeyboard('{safe_label}')")
        elif self.mode == "eye":
            if label == "switch_to_keyboard":
                self.enter_keyboard_mode()
            elif label == "left_click":
                self.mouse.click(Button.left)
            elif label == "right_click":
                self.mouse.click(Button.right)
        # mode == "setup": gestures don't drive anything here - the
        # calibration UI itself is button-driven, not gesture-driven.

    # --- called from the keyboard window's js_api (the on-screen "toggle" key) --

    def enter_eye_mode(self) -> None:
        """The keyboard's own "toggle" key: the *only* way out of
        keyboard mode, deliberately not a gesture."""
        self.keyboard_window.hide()
        self._set_mode("eye")

    def enter_keyboard_mode(self) -> None:
        """The eye-mode "switch_to_keyboard" gesture: the *only* way
        back from eye mode, deliberately not a button (there's no
        keyboard visible to click one on)."""
        show_without_stealing_focus(self.keyboard_window)
        self._set_mode("keyboard")

    # --- called from the toggle tab's js_api ---------------------------

    def reopen_calibration(self) -> None:
        """Left-edge tab: back to setup from either mode."""
        self.toggle_window.hide()
        self.keyboard_window.hide()
        self.main_window.show()
        self._set_mode("setup")
