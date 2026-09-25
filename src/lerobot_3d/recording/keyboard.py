"""Single-key terminal input without Enter-to-submit or echo (POSIX ``termios``).

No extra dependency and no X display: keys are read from the controlling terminal, so the
terminal running the program must have focus.
"""
from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import tty


class TerminalKeyListener:
    """Context manager: puts stdin into cbreak mode and queues each key pressed.

    Ctrl+C still raises ``KeyboardInterrupt``; the terminal is restored on exit either way.
    """

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._keys: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._saved_attrs = None

    def __enter__(self) -> "TerminalKeyListener":
        if not self._stream.isatty():
            raise RuntimeError(
                "Keyboard controls need an interactive terminal (stdin is not a TTY)."
            )
        self._fd = self._stream.fileno()
        self._saved_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._fd is not None and self._saved_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if ready:
                data = os.read(self._fd, 32)
                for ch in data.decode(errors="ignore"):
                    self._keys.put(ch)

    def get_keys(self) -> list[str]:
        """All keys pressed since the last call (non-blocking)."""
        keys = []
        while True:
            try:
                keys.append(self._keys.get_nowait())
            except queue.Empty:
                return keys
