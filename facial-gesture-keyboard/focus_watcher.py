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
UIA_GROUP_CONTROL_TYPE_ID = 50026
UIA_PANE_CONTROL_TYPE_ID = 50033
UIA_CUSTOM_CONTROL_TYPE_ID = 50025

# Edit and ComboBox mean "typeable" unconditionally - every native and
# browser text input reports one of these, no further check needed.
ALWAYS_TYPEABLE_CONTROL_TYPES = {
    UIA_EDIT_CONTROL_TYPE_ID,
    UIA_COMBO_BOX_CONTROL_TYPE_ID,
}

# Document, on the other hand, is genuinely ambiguous: a native app's
# editable body (Word) *and* a browser's read-only page content both
# report as Document - confirmed live that Chrome moves focus to the
# page's root "Document" element on every full navigation (a known
# behavior for screen-reader compatibility), which is exactly what was
# popping the keyboard up on every link click/new page with nothing
# typeable on it. Group/Pane/Custom are the reverse case: confirmed live
# that a real contenteditable region (the kind of hidden capture surface
# rich editors like Google Docs use) reports as Group, not Document, so
# it was never being detected as typeable at all. Both directions are
# resolved by the same check - see _is_readonly_text() below - rather
# than trusting or excluding control type alone.
DOCUMENT_LIKE_CONTROL_TYPES = {UIA_DOCUMENT_CONTROL_TYPE_ID}
CONTAINER_CONTROL_TYPES_NEEDING_PROOF = {
    UIA_GROUP_CONTROL_TYPE_ID,
    UIA_PANE_CONTROL_TYPE_ID,
    UIA_CUSTOM_CONTROL_TYPE_ID,
}

FOCUS_POLL_INTERVAL_S = 0.3

# A typeable reading has to hold up across this many consecutive polls
# on the *same* element before the keyboard actually opens - a full page
# navigation (clicking a link, landing on a new page) briefly moves OS
# focus through intermediate states while the page is still settling,
# and a transient element that happens to look typeable for one poll
# only should not be enough to pop the keyboard up on a page with
# nothing typeable on it. Genuinely clicking into a real field holds
# "typeable" for far longer than this, so real use is unaffected -
# closing is intentionally NOT debounced (see _run below), so leaving a
# field still dismisses the keyboard immediately.
FOCUS_OPEN_CONFIRM_POLLS = 2


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
        self._last_typeable: bool | None = None  # tracks the *confirmed* (debounced) state
        self._last_element_id: tuple | None = None
        self._pending_element_id: tuple | None = None
        self._pending_count: int = 0
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
                is_typeable, element_id = self._inspect_focused_element(uia, uia_module)
            except Exception:
                is_typeable, element_id = False, None

            # Require a typeable reading to hold up for
            # FOCUS_OPEN_CONFIRM_POLLS consecutive polls on the *same*
            # element before treating it as confirmed - filters out a
            # one-poll-only transient (e.g. mid-navigation) without
            # requiring genuine field clicks to wait any meaningful
            # amount of time (they hold "typeable" far longer than this).
            if is_typeable and element_id == self._pending_element_id:
                self._pending_count += 1
            elif is_typeable:
                self._pending_element_id = element_id
                self._pending_count = 1
            else:
                self._pending_element_id = None
                self._pending_count = 0

            confirmed_typeable = is_typeable and self._pending_count >= FOCUS_OPEN_CONFIRM_POLLS

            newly_typeable_target = confirmed_typeable and (
                not self._last_typeable or element_id != self._last_element_id
            )
            # Closing is intentionally based on the raw (non-debounced)
            # reading, not the confirmed one - once a field really is
            # focused, leaving it should dismiss the keyboard right away,
            # with no added delay.
            left_typeable = not is_typeable and self._last_typeable

            if newly_typeable_target:
                self._on_typeable_changed(True)
            elif left_typeable:
                self._on_typeable_changed(False)

            self._last_typeable = confirmed_typeable
            self._last_element_id = element_id if confirmed_typeable else None

            time.sleep(FOCUS_POLL_INTERVAL_S)

    @staticmethod
    def _is_readonly_text(element, uia_module) -> bool | None:
        """True/False if the element's text content is provably
        read-only/editable, None if that can't be determined (no Text
        pattern, or the IsReadOnly text attribute isn't implemented for
        this element) - callers fall back to their own default rather
        than trusting an inconclusive result.

        Verified live against three real cases before relying on this:
        Word's document body -> False (editable), a plain Wikipedia
        article's root web area -> True (read-only), a real
        contenteditable region -> False (editable). Value/TextEdit
        pattern *availability* looked like promising signals at first but
        turned out to just reflect the framework (Chrome reports both as
        available on everything, Word on neither) rather than actual
        editability - this text attribute was the one property that came
        back correct in all three cases.
        """
        try:
            has_text_pattern = bool(
                element.GetCurrentPropertyValue(uia_module.UIA_IsTextPatternAvailablePropertyId)
            )
            if not has_text_pattern:
                return None
            text_pattern = element.GetCurrentPattern(uia_module.UIA_TextPatternId)
            text_pattern = text_pattern.QueryInterface(uia_module.IUIAutomationTextPattern)
            value = text_pattern.DocumentRange.GetAttributeValue(uia_module.UIA_IsReadOnlyAttributeId)
        except Exception:
            return None
        return value if isinstance(value, bool) else None

    @classmethod
    def _inspect_focused_element(cls, uia, uia_module) -> tuple[bool, tuple | None]:
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

        control_type = element.CurrentControlType
        if control_type in ALWAYS_TYPEABLE_CONTROL_TYPES:
            is_typeable = True
        elif control_type in DOCUMENT_LIKE_CONTROL_TYPES:
            # Trusted by default (this is what made Word/native document
            # editors work in the first place) - only overridden when we
            # can *prove* the content is read-only.
            is_typeable = cls._is_readonly_text(element, uia_module) is not True
        elif control_type in CONTAINER_CONTROL_TYPES_NEEDING_PROOF:
            # Excluded by default (these are common, generic container
            # types used all over every UI) - only trusted when we can
            # *prove* the content is genuinely editable.
            is_typeable = cls._is_readonly_text(element, uia_module) is False
        else:
            is_typeable = False

        if not is_typeable:
            return False, None
        try:
            element_id = element.GetRuntimeId()
        except Exception:
            element_id = None
        return True, element_id
