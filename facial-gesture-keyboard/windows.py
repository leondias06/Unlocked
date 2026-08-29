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
                 mouse clicks). Entered from "setup" via the Confirm
                 button - this, not the keyboard, is the resting state
                 after confirming: you land able to control the cursor,
                 and the keyboard comes up on demand, rather than the
                 other way around.
  - "keyboard" - the keyboard overlay is visible and receives
                 up/down/left/right/confirm/backspace gestures. Entered
                 from "eye" *automatically*, not via a gesture - see
                 focus_watcher.py: whenever the OS-focused control
                 anywhere on the system (a browser's search bar, a Word
                 document, Notepad, ...) is something you can actually
                 type into, the keyboard comes up, the same way a
                 phone's on-screen keyboard appears when you tap a text
                 field. There used to be a switch_to_keyboard gesture
                 for this; it was retired because it shared a near-
                 identical calibrated movement with left_click, and if a
                 stray confirm ever landed on the keyboard's own
                 "toggle" key first (dropping back to eye mode without
                 the user noticing), every further attempt at that
                 movement would then correctly - for eye mode - fire as
                 a real mouse click, which is confusing to debug and
                 only shows up as "gestures feel broken" downstream.
                 Auto-detecting focus sidesteps the whole failure mode:
                 there's no gesture to misfire in the first place. The
                 *only* way back out to eye mode is the on-screen
                 "toggle" key inside the keyboard grid itself (a normal
                 navigate+confirm key press, not a gesture).

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
from focus_watcher import FocusWatcher

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

    def __init__(
        self,
        main_window: webview.Window,
        keyboard_window: webview.Window,
        toggle_window: webview.Window,
        debug_window: webview.Window | None = None,
    ) -> None:
        self.main_window = main_window
        self.keyboard_window = keyboard_window
        self.toggle_window = toggle_window
        self.debug_window = debug_window
        self.mode = "setup"
        self.mouse = MouseController()
        server_module.set_active_mode(self.mode)
        self.focus_watcher = FocusWatcher(self._on_focus_typeable_changed)
        self.focus_watcher.start()

    def _on_focus_typeable_changed(self, is_typeable: bool) -> None:
        """FocusWatcher callback (runs on its own background thread): the
        OS-focused control *anywhere on the system* just became typeable,
        or stopped being. Only acts in eye/keyboard mode - during setup
        there's no keyboard to bring up yet regardless of what's focused
        elsewhere."""
        if is_typeable and self.mode == "eye":
            self.enter_keyboard_mode()
        elif not is_typeable and self.mode == "keyboard":
            self.enter_eye_mode()

    @staticmethod
    def _js_safe(s: str) -> str:
        return s.replace("\\", "").replace("'", "")

    def update_live_gesture(self, prediction: str | None, confidence: float) -> None:
        """Called every live-classified frame (see MainApi.update_live_gesture)
        to drive the small always-on-top debug overlay - "just for testing
        purposes", per spec, so it deliberately shows the *raw* per-frame
        reading, not the debounced/mode-gated result."""
        if self.debug_window is None:
            return
        label = self._js_safe(prediction or "neutral")
        pct = round((confidence or 0) * 100)
        self.debug_window.evaluate_js(f"window.updateLive?.('{self.mode}', '{label}', {pct})")

    def update_cursor_debug(self, ready: bool, yaw_delta: float, pitch_delta: float, moving: bool) -> None:
        """Live head-pose deflection from the eye-mode cursor's centered
        baseline - see main.py's cursor_debug. Makes drift ("cursor won't
        stop moving up") diagnosable: if it's sitting well outside the
        dead zone even when you believe you're holding still, the
        baseline itself is probably off, not the dead zone or gesture
        recognition. `moving` is main.py's own authoritative dead-zone
        check (also what gates left_click/right_click - see
        cursor_is_moving there), not re-derived here, so the overlay
        can't disagree with what's actually happening."""
        if self.debug_window is None:
            return
        self.debug_window.evaluate_js(
            f"window.updateCursorDebug?.({str(bool(ready)).lower()}, {yaw_delta}, {pitch_delta}, {str(bool(moving)).lower()})"
        )

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
        control and bringing the keyboard up automatically once you
        focus something typeable (see FocusWatcher) is."""
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
        if self.debug_window is not None:
            # Logged before mode-gating below, deliberately - a gesture
            # silently ignored because it's the "wrong" mode still shows
            # up here, which is exactly what you need to catch something
            # firing when you didn't mean it to.
            self.debug_window.evaluate_js(
                f"window.flashFired?.('{self.mode}', '{self._js_safe(label)}')"
            )

        if self.mode == "keyboard":
            if label in KEYBOARD_MODE_GESTURES:
                safe_label = self._js_safe(label)
                self.keyboard_window.evaluate_js(f"applyGestureToKeyboard('{safe_label}')")
        elif self.mode == "eye":
            if label == "left_click":
                self.mouse.click(Button.left)
            elif label == "right_click":
                self.mouse.click(Button.right)
        # mode == "setup": gestures don't drive anything here - the
        # calibration UI itself is button-driven, not gesture-driven.

    # --- called from the keyboard window's js_api (the on-screen "toggle" key) --

    def enter_eye_mode(self) -> None:
        """Two ways in: the keyboard's own "toggle" key (manual,
        deliberately not a gesture), or FocusWatcher noticing the
        OS-focused control stopped being typeable (automatic, e.g. you
        clicked away from the text field entirely)."""
        self.keyboard_window.hide()
        self._set_mode("eye")

    def enter_keyboard_mode(self) -> None:
        """Called by FocusWatcher when the OS-focused control becomes
        typeable - by design, that's the only way into keyboard mode;
        there's no manual button or gesture for this direction."""
        show_without_stealing_focus(self.keyboard_window)
        self._set_mode("keyboard")

    # --- called from the toggle tab's js_api ---------------------------

    def reopen_calibration(self) -> None:
        """Left-edge tab: back to setup from either mode."""
        self.toggle_window.hide()
        self.keyboard_window.hide()
        self.main_window.show()
        self._set_mode("setup")
