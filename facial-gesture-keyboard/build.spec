# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for the standalone desktop app.

Build with (from this directory, inside the venv):
    pyinstaller build.spec

Produces dist/FacialGestureKeyboard.exe - a single file, no Python
install required to run it, no network access needed at runtime (the
MediaPipe model is bundled in rather than downloaded on first run).
"""

from PyInstaller.utils.hooks import collect_all

datas = [
    ("static", "static"),
    ("face_landmarker.task", "."),
]
binaries = []
hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

# mediapipe, pynput and webview all load platform-specific native
# binaries/resources dynamically rather than through plain Python
# imports PyInstaller's static analysis can see - collect_all grabs
# everything each package ships rather than guessing. comtypes is here
# for a different reason: focus_watcher.py's UI Automation bindings
# (comtypes.gen.UIAutomationClient) are generated from a COM type
# library the *first time* comtypes.client.GetModule runs, then cached
# as plain .py files under comtypes/gen/ - that generation already
# happened once in dev (see focus_watcher.py's docstring), so by build
# time those files already exist on disk and just need to be bundled
# like any other module, not regenerated inside the frozen .exe (whose
# comtypes/gen/ directory - inside the onefile temp extraction - isn't
# a stable place to cache anything across runs anyway).
for pkg in ("mediapipe", "pynput", "webview", "comtypes"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FacialGestureKeyboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
