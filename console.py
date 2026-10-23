"""Get our own console window out of the way while the screen is captured."""
import contextlib
import ctypes
import sys
import time


def _console_hwnd():
    if sys.platform != "win32":
        return None
    k32, u32 = ctypes.windll.kernel32, ctypes.windll.user32
    k32.GetConsoleWindow.restype = ctypes.c_void_p
    u32.GetAncestor.restype = ctypes.c_void_p
    u32.GetAncestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
    hwnd = k32.GetConsoleWindow()
    if not hwnd:
        return None
    # Windows Terminal hosts consoles behind a hidden pseudo-window owned by the real one
    owner = u32.GetAncestor(hwnd, 3)          # GA_ROOTOWNER
    return owner or hwnd


@contextlib.contextmanager
def hidden(delay=0.5):
    """Minimise the console for the duration of the block, then bring it back."""
    hwnd = _console_hwnd()
    if not hwnd:
        yield
        return
    u32 = ctypes.windll.user32
    u32.ShowWindow.argtypes = (ctypes.c_void_p, ctypes.c_int)
    u32.SetForegroundWindow.argtypes = (ctypes.c_void_p,)
    u32.ShowWindow(hwnd, 6)                   # SW_MINIMIZE
    time.sleep(delay)                         # let the minimise animation finish
    try:
        yield
    finally:
        u32.ShowWindow(hwnd, 9)               # SW_RESTORE
        # tapping Alt lets a background process take the foreground back
        u32.keybd_event(0x12, 0, 0, 0)
        u32.keybd_event(0x12, 0, 2, 0)
        u32.SetForegroundWindow(hwnd)
