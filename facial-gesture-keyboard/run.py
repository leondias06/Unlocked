"""
Launches the dev server with --reload safely.

Why this exists instead of just `uvicorn main:app --reload`: uvicorn's
default file watcher recurses through the *entire* project directory,
including .venv - which contains thousands of files from mediapipe,
scipy, etc. Anything that touches their timestamps (pip, antivirus,
search indexing) looks like a code change and restarts the whole
server, wiping in-memory state (this is what was causing calibration
progress to reset) and occasionally crashing with a BrokenPipeError if
a restart lands mid-restart.

`--reload-exclude` on the command line can't fix this on Windows: Click
(uvicorn's CLI framework) auto-expands any `*` in a command-line
argument against real files before uvicorn ever sees it, so
`--reload-exclude ".venv/*"` silently turns into several unrelated
literal filenames and the command fails to parse. Calling uvicorn.run()
directly from Python sidesteps that entirely - the exclude path never
goes through argv/Click, so we can pass it as a real, absolute path
with no globbing surprises.
"""

from pathlib import Path

import uvicorn

if __name__ == "__main__":
    venv_dir = Path(__file__).parent / ".venv"
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=[str(venv_dir)],
    )
