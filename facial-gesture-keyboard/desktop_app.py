"""
Desktop app entry point - what actually gets packaged into the
standalone .exe (see build.spec).

Runs the exact same FastAPI app as `main.py` (no --reload; a packaged
app doesn't need to watch for source edits) in a background thread, and
opens it in a native window via pywebview instead of a browser tab. No
network access is required at runtime: the MediaPipe model file is
bundled into the build rather than downloaded on first run (see
build.spec and main.ensure_model()).
"""

import socket
import threading
import time
import urllib.request

import uvicorn
import webview

import main as server_module


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_server(port: int) -> None:
    uvicorn.run(server_module.app, host="127.0.0.1", port=port, log_level="warning")


def wait_until_up(port: int, timeout: float = 20.0) -> None:
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


def main() -> None:
    port = find_free_port()
    threading.Thread(target=run_server, args=(port,), daemon=True).start()
    wait_until_up(port)

    webview.create_window(
        "Facial Gesture Keyboard",
        url=f"http://127.0.0.1:{port}/",
        width=1280,
        height=860,
        min_size=(900, 640),
    )
    webview.start()


if __name__ == "__main__":
    main()
