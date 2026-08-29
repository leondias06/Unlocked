"""
Desktop app entry point - what actually gets packaged into the
standalone .exe (see build.spec).

Runs the exact same FastAPI app as `main.py` (no --reload; a packaged
app doesn't need to watch for source edits) in a background thread, and
opens it across three native windows via pywebview instead of a browser
tab:
  - the launch/calibration page (camera + calibration UI + Confirm)
  - the on-screen keyboard, an always-on-top overlay (see windows.py
    for how it avoids stealing OS keyboard focus from whatever app
    you're actually typing into) - shown by Confirm, hidden by its own
    "toggle" key (switches to eye/mouse mode), shown again by that
    mode's "switch_to_keyboard" gesture
  - a small tab docked to the left screen edge that reopens calibration
    from either mode

See windows.DesktopWindows for the actual mode state machine
(setup / keyboard / eye) - this file just creates the windows and
wires each one's js_api to it.

No network access is required at runtime: the MediaPipe model file is
bundled into the build rather than downloaded on first run (see
build.spec and main.ensure_model()).
"""

import ctypes
import socket
import sys
import threading
import time
import traceback
import urllib.request

import uvicorn
import webview

import main as server_module
from windows import DesktopWindows

# Windows enforces its own minimum window width (~120px) regardless of
# what's requested here or via min_size - narrower values just get
# silently clamped up to that floor. 120 is as slim as this tab gets;
# height has no such floor, so it's kept short to look like a small tab.
TOGGLE_WIDTH = 120
TOGGLE_HEIGHT = 64
KEYBOARD_WIDTH = 720
KEYBOARD_HEIGHT = 480


class MainApi:
    """
    js_api for the main (calibration) window.

    pywebview auto-exposes every public (non-underscore) attribute of a
    js_api object to JS, recursively walking into any non-callable
    object it finds to build the bridge's function map. `_windows`
    (the DesktopWindows instance) holds real webview.Window objects
    wrapping WinForms/.NET COM objects - if it weren't
    underscore-prefixed, that walk would recurse into
    `.native.AccessibilityObject.Bounds`, hit .NET's self-referential
    `Rectangle.Empty` static property, and recurse effectively forever,
    pegging a CPU core hard enough to make the whole app (including the
    HTTP/WebSocket server, which shares the process) unresponsive.
    """

    def __init__(self) -> None:
        self._windows: DesktopWindows | None = None

    def confirm_calibration(self) -> None:
        self._windows.confirm_calibration()

    def on_gesture(self, label: str) -> None:
        self._windows.on_gesture(label)


class ToggleApi:
    """js_api for the small left-edge toggle tab. See MainApi's
    docstring for why this attribute must be underscore-prefixed."""

    def __init__(self) -> None:
        self._windows: DesktopWindows | None = None

    def reopen_calibration(self) -> None:
        self._windows.reopen_calibration()


class KeyboardApi:
    """js_api for the keyboard window. See MainApi's docstring for why
    this attribute must be underscore-prefixed."""

    def __init__(self) -> None:
        self._windows: DesktopWindows | None = None

    def enter_eye_mode(self) -> None:
        self._windows.enter_eye_mode()


def bind_free_socket() -> socket.socket:
    """
    Binds and listens on an OS-assigned port, returning the live socket
    itself rather than just the port number. uvicorn is then handed this
    exact socket to serve on - a "find a free port, close it, then bind
    that port number again" approach has a race where something else
    can grab the port in between; reusing the same socket object has no
    such window.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    return sock


def run_server(sock: socket.socket) -> None:
    config = uvicorn.Config(server_module.app, log_level="warning")
    uvicorn.Server(config).run(sockets=[sock])


def wait_until_up(port: int, timeout: float = 45.0) -> None:
    # Generous timeout: a packaged .exe's first launch has to extract
    # ~200MB to a temp dir and cold-import mediapipe/opencv, which can
    # take a while on a slower machine or one with active AV scanning.
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Server did not start within {timeout}s") from last_error


def show_fatal_error(message: str) -> None:
    """
    build.spec runs this windowed (console=False), so an uncaught
    exception normally has nowhere to go - the window just vanishes
    with no explanation. This puts the error somewhere the person
    running the app can actually see it.
    """
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        MB_ICONERROR = 0x10
        ctypes.windll.user32.MessageBoxW(None, message, "Facial Gesture Keyboard - failed to start", MB_ICONERROR)


def screen_size() -> tuple[int, int]:
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def main() -> None:
    try:
        sock = bind_free_socket()
        port = sock.getsockname()[1]
        threading.Thread(target=run_server, args=(sock,), daemon=True).start()
        wait_until_up(port)
    except Exception:
        show_fatal_error("Failed to start the local server:\n\n" + traceback.format_exc())
        return

    screen_w, screen_h = screen_size()
    base_url = f"http://127.0.0.1:{port}"

    main_api = MainApi()
    toggle_api = ToggleApi()
    keyboard_api = KeyboardApi()

    main_window = webview.create_window(
        "Facial Gesture Keyboard",
        url=f"{base_url}/",
        width=1280,
        height=860,
        min_size=(900, 640),
        js_api=main_api,
    )

    keyboard_window = webview.create_window(
        "Keyboard",
        url=f"{base_url}/keyboard.html",
        width=KEYBOARD_WIDTH,
        height=KEYBOARD_HEIGHT,
        x=(screen_w - KEYBOARD_WIDTH) // 2,
        y=screen_h - KEYBOARD_HEIGHT - 60,
        frameless=True,
        on_top=True,
        hidden=True,
        js_api=keyboard_api,
    )

    toggle_window = webview.create_window(
        "Toggle",
        url=f"{base_url}/toggle.html",
        width=TOGGLE_WIDTH,
        height=TOGGLE_HEIGHT,
        x=0,
        y=(screen_h - TOGGLE_HEIGHT) // 2,
        frameless=True,
        on_top=True,
        hidden=True,
        js_api=toggle_api,
        # pywebview defaults min_size to (200, 100), which silently
        # clamped this to a much wider box than intended - it's meant
        # to be a slim sliver hanging off the screen edge.
        min_size=(TOGGLE_WIDTH, TOGGLE_HEIGHT),
    )

    desktop_windows = DesktopWindows(main_window, keyboard_window, toggle_window)
    main_api._windows = desktop_windows
    toggle_api._windows = desktop_windows
    keyboard_api._windows = desktop_windows

    webview.start()


if __name__ == "__main__":
    main()
