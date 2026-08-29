"""
Detects whether the OS-wide focused UI element - in *any* app, not just
this one - is something you can type into (a text box, a document, a
combo box with an editable field), so the keyboard overlay can come up
automatically the way a phone's on-screen keyboard does: show up when
you tap a field, go away when you tap elsewhere. Replaces the old
switch_to_keyboard *gesture* entirely - there's nothing to calibrate or
misfire here, since it's driven by the same OS focus state any assistive
tech or accessibility tool already relies on.

Uses Windows UI Automation (UIA) via comtypes, polling the focused
element every FOCUS_POLL_INTERVAL_S. UIA is the right tool for this
specifically because it works across different UI frameworks (native
Win32 edit controls, Word, and - since Chrome/Edge/Firefox all implement
UIA for their own accessibility support - browser text fields like a
Google search bar) rather than only recognizing one app's native
controls. The trade-off: some custom-rendered apps (certain Electron
apps, some games, canvas-based editors) have incomplete UIA support and
may not be detected - there's no gesture fallback for that class of app
right now.
"""

from __future__ import annotations

import os
import threading
import time

import comtypes
import comtypes.client

_OWN_PID = os.getpid()

# Stable, documented numeric UIA control-type IDs - hardcoded rather
# than pulled from the generated module's constants, so this doesn't
# depend on exactly how comtypes names them across versions.
UIA_EDIT_CONTROL_TYPE_ID = 50004
UIA_COMBO_BOX_CONTROL_TYPE_ID = 50003
UIA_DOCUMENT_CONTROL_TYPE_ID = 50030
TYPEABLE_CONTROL_TYPES = {
    UIA_EDIT_CONTROL_TYPE_ID,
    UIA_COMBO_BOX_CONTROL_TYPE_ID,
    UIA_DOCUMENT_CONTROL_TYPE_ID,
}

FOCUS_POLL_INTERVAL_S = 0.3


def _load_uia_module():
    # Generates (and caches, in comtypes.gen) Python bindings from the
    # UIA type library on first call - see build.spec for how this is
    # pre-generated and bundled so the frozen .exe doesn't need to
    # regenerate it at runtime.
    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient as UIA  # noqa: PLC0415
    return UIA


class FocusWatcher:
    """Runs a background polling thread; calls on_typeable_changed(bool).

    Tracks the focused *element's identity* (UIA's GetRuntimeId(), a
    stable per-element key), not just a typeable/not-typeable boolean -
    that distinction matters a lot in practice. A pure boolean edge
    trigger misses the common case of clicking from one typeable field
    straight into another (a browser's address bar into its search box,
    say): "typeable" was already true and stays true, so nothing would
    ever re-fire, and the keyboard just wouldn't come up for that click -
    this is what made pop-up feel inconsistent rather than simply
    "doesn't work in certain apps". Firing again whenever the specific
    focused element changes, even between two typeable elements, fixes
    that. The one remaining case that still won't re-trigger is manually
    dismissing the keyboard (the on-screen "toggle" key) while staying on
    the exact same field, then wanting it back without changing focus at
    all - by design, since nothing about the focus state actually changed
    for UIA to observe.
    """

    def __init__(self, on_typeable_changed) -> None:
        self._on_typeable_changed = on_typeable_changed
        self._last_typeable: bool | None = None
        self._last_element_id: tuple | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        comtypes.CoInitialize()
        try:
            uia_module = _load_uia_module()
            uia = comtypes.client.CreateObject(
                uia_module.CUIAutomation, interface=uia_module.IUIAutomation
            )
        except Exception as exc:
            # Best-effort feature: if UI Automation isn't available for
            # some reason, the app should keep working via the manual
            # on-screen "toggle" key rather than crash the whole thing.
            print(f"[focus_watcher] UI Automation unavailable, auto-detect disabled: {exc}")
            return

        while not self._stop.is_set():
            try:
                is_typeable, element_id = self._inspect_focused_element(uia)
            except Exception:
                is_typeable, element_id = False, None

            newly_typeable_target = is_typeable and (
                not self._last_typeable or element_id != self._last_element_id
            )
            left_typeable = not is_typeable and self._last_typeable

            if newly_typeable_target:
                self._on_typeable_changed(True)
            elif left_typeable:
                self._on_typeable_changed(False)

            self._last_typeable = is_typeable
            self._last_element_id = element_id if is_typeable else None

            time.sleep(FOCUS_POLL_INTERVAL_S)

    @staticmethod
    def _inspect_focused_element(uia) -> tuple[bool, tuple | None]:
        element = uia.GetFocusedElement()
        if element is None:
            return False, None
        if element.CurrentProcessId == _OWN_PID:
            # The keyboard overlay's own <textarea> preview is a real,
            # focusable Edit-type control - if OS focus ever lands on
            # one of *our own* windows (observed happening when an
            # external app closes while the overlay is topmost; it
            # shouldn't per WS_EX_NOACTIVATE, but Windows' automatic
            # focus-reassignment on window close doesn't always respect
            # that), it must never count as "you focused something
            # typeable" - that would be reacting to our own UI, not
            # anything the user actually did.
            return False, None
        if element.CurrentControlType not in TYPEABLE_CONTROL_TYPES:
            return False, None
        try:
            element_id = element.GetRuntimeId()
        except Exception:
            element_id = None
        return True, element_id
