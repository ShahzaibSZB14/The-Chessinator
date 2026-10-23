"""Mouse backends.

"sendinput" (Windows default) injects real mouse events with SendInput, the same
path a physical mouse takes. Games and apps that read raw input (and ignore a
cursor that just teleports via SetCursorPos) see genuine moves, presses and
drags. "pynput" is the portable fallback for other platforms.
"""
import ctypes
import sys
import time

from pynput import mouse as _pmouse


class PynputMouse:
    def __init__(self):
        self.c = _pmouse.Controller()

    @property
    def position(self):
        return self.c.position

    def move(self, x, y):
        self.c.position = (int(round(x)), int(round(y)))

    def down(self):
        self.c.press(_pmouse.Button.left)

    def up(self):
        self.c.release(_pmouse.Button.left)


if sys.platform == "win32":
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_size_t

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]

    class _INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

    _MOVE, _LEFTDOWN, _LEFTUP = 0x0001, 0x0002, 0x0004
    _VIRTUALDESK, _ABSOLUTE = 0x4000, 0x8000

    class SendInputMouse:
        def __init__(self):
            self.u32 = ctypes.windll.user32
            self.u32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)

        def _send(self, flags, dx=0, dy=0):
            inp = _INPUT(type=0, u=_INPUTUNION(mi=_MOUSEINPUT(dx, dy, 0, flags, 0, 0)))
            self.u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

        @property
        def position(self):
            pt = wintypes.POINT()
            self.u32.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)

        def move(self, x, y):
            vx, vy = self.u32.GetSystemMetrics(76), self.u32.GetSystemMetrics(77)
            vw, vh = self.u32.GetSystemMetrics(78), self.u32.GetSystemMetrics(79)
            nx = int(round((x - vx) * 65535 / max(1, vw - 1)))
            ny = int(round((y - vy) * 65535 / max(1, vh - 1)))
            self._send(_MOVE | _ABSOLUTE | _VIRTUALDESK, nx, ny)

        def down(self):
            self._send(_LEFTDOWN)

        def up(self):
            self._send(_LEFTUP)


def make_mouse(kind="auto"):
    if kind == "auto":
        kind = "sendinput" if sys.platform == "win32" else "pynput"
    return SendInputMouse() if kind == "sendinput" else PynputMouse()


class Hand:
    """Human-paced mouse actions built on a backend; timings suit apps that run
    their UI logic once per frame (they must *see* the hover/press/motion)."""

    def __init__(self, backend, settle=0.05, hold=0.04, drag_time=0.2, click_gap=0.08):
        self.m = backend
        self.settle, self.hold, self.drag_time, self.click_gap = settle, hold, drag_time, click_gap

    @property
    def position(self):
        return self.m.position

    def glide(self, a, b, duration):
        """Move in steps so every frame in between sees the cursor travel."""
        steps = max(4, int(duration / 0.012))
        for i in range(1, steps + 1):
            t = i / steps
            self.m.move(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            time.sleep(duration / steps)

    def hover(self, p):
        """Arrive at p with a tiny wiggle so hover handlers fire."""
        self.m.move(p[0] + 2, p[1] + 2)
        time.sleep(0.01)
        self.m.move(p[0], p[1])
        time.sleep(self.settle)

    def click(self, p):
        self.hover(p)
        self.m.down()
        time.sleep(self.hold)
        self.m.up()

    def drag(self, a, b):
        self.hover(a)
        self.m.down()
        time.sleep(self.hold)
        # nudge first: many UIs only start a drag after a few pixels of motion
        mid = (a[0] + (b[0] - a[0]) * 0.1, a[1] + (b[1] - a[1]) * 0.1)
        self.glide(a, mid, self.drag_time * 0.2)
        self.glide(mid, b, self.drag_time * 0.8)
        self.m.move(b[0], b[1])
        time.sleep(self.settle)
        self.m.up()

    def move_to(self, p):
        self.m.move(p[0], p[1])
