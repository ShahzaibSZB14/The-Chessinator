"""A see-through overlay drawn on top of your real board (Windows).

It is a borderless, always-on-top, click-through window that covers the whole
virtual desktop. "Click-through" means every mouse event passes straight to the
window underneath, so the overlay never gets in the way of playing - you click
the real board as usual and only ever *see* the arrow and the evaluation bar.

The overlay runs in its OWN process (this file, launched with --serve), with Tk
on that process's main thread. The analysis program talks to it by writing one
JSON line per frame to the child's stdin. A separate process is used on purpose:
Tk does not survive being driven from a second thread of the main program (Tcl
aborts at shutdown with "async handler deleted by the wrong thread"), whereas a
child process is simply terminated when we're done. If the child can't start,
Overlay.ok is False and the caller falls back to the console.
"""
import ctypes
import json
import subprocess
import sys
import threading

TRANSPARENT = "#010101"                 # this exact colour is punched out to see-through
ARROW = "#17c964"                       # best-move arrow (green)
ARROW_EDGE = "#0a3d20"
BAR_WHITE = "#f2f2f2"
BAR_BLACK = "#2b2b2b"
PANEL_BG = "#101418"
TEXT = "#eaeaea"
MUTED = "#8b98a5"

# Window styles for a layered, click-through window.
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080           # keep it out of the taskbar / alt-tab


def _virtual_desktop():
    """(left, top, width, height) of the whole virtual desktop in real pixels."""
    u = ctypes.windll.user32
    return (u.GetSystemMetrics(76), u.GetSystemMetrics(77),
            u.GetSystemMetrics(78), u.GetSystemMetrics(79))


# ==========================================================================
# Parent side: a thin handle that owns the child process
# ==========================================================================

class Overlay:
    def __init__(self):
        self.ok = False
        self.proc = None
        if sys.platform != "win32":
            return
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-I", __file__, "--serve"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except Exception:
            self.proc = None
            return
        self.ok = self._await_ready(timeout=6.0)
        if not self.ok:
            self.close()

    def _await_ready(self, timeout):
        """The child prints READY once its window exists (FAIL if Tk won't start)."""
        result = {}

        def reader():
            line = self.proc.stdout.readline()
            result["line"] = line.strip()
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout)
        return result.get("line") == "READY"

    def set_state(self, quad=None, arrow=None, eval_white=0.0, mate=None,
                  white_bottom=True, lines=(), status=""):
        """Send one frame. Coordinates are absolute screen pixels."""
        if not self.ok or self.proc is None or self.proc.poll() is not None:
            return
        msg = {
            "quad": None if quad is None else [[float(x), float(y)] for x, y in quad],
            "arrow": None if arrow is None else [float(v) for v in arrow],
            "eval_white": float(eval_white),
            "mate": None if mate is None else int(mate),
            "white_bottom": bool(white_bottom),
            "lines": list(lines),
            "status": str(status),
        }
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except Exception:
            self.ok = False

    def clear(self):
        self.set_state(quad=None, arrow=None, lines=(), status="")

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2.0)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None
        self.ok = False


# ==========================================================================
# Child side: the actual Tk window (runs when launched with --serve)
# ==========================================================================

class _Window:
    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self._lock = threading.Lock()
        self._state = None
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", TRANSPARENT)
        left, top, w, h = _virtual_desktop()
        self.origin = (left, top)
        self.root.geometry(f"{w}x{h}+{left}+{top}")
        self.canvas = tk.Canvas(self.root, bg=TRANSPARENT, highlightthickness=0,
                                borderwidth=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._make_click_through()

    def _make_click_through(self):
        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        u = ctypes.windll.user32
        getfn = getattr(u, "GetWindowLongPtrW", u.GetWindowLongW)
        setfn = getattr(u, "SetWindowLongPtrW", u.SetWindowLongW)
        style = getfn(hwnd, GWL_EXSTYLE)
        setfn(hwnd, GWL_EXSTYLE,
              style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)

    # -- stdin reader (background thread; Tk stays on the main thread) ----
    def serve(self):
        threading.Thread(target=self._read_stdin, daemon=True).start()
        self.root.after(60, self._tick)
        self.root.mainloop()

    def _read_stdin(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                state = json.loads(line)
            except Exception:
                continue
            with self._lock:
                self._state = state
        # stdin closed -> parent is done with us
        try:
            self.root.after(0, self.root.quit)
        except Exception:
            pass

    def _tick(self):
        with self._lock:
            state = self._state
        try:
            self._redraw(state)
        except Exception:
            pass
        self.root.after(60, self._tick)

    # -- drawing ---------------------------------------------------------
    def _to_local(self, x, y):
        return x - self.origin[0], y - self.origin[1]

    def _redraw(self, state):
        c = self.canvas
        c.delete("all")
        if not state or state.get("quad") is None:
            return
        quad = [self._to_local(x, y) for x, y in state["quad"]]
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        board_w, board_h = bx1 - bx0, by1 - by0
        sq = max(12.0, min(board_w, board_h) / 8.0)
        self._draw_eval_bar(c, bx0, by0, board_h, sq, state)
        if state.get("arrow"):
            self._draw_arrow(c, state["arrow"], sq)
        self._draw_panel(c, bx0, by1, board_w, state)

    def _draw_eval_bar(self, c, bx0, by0, board_h, sq, state):
        bw = max(10.0, sq * 0.32)
        gap = sq * 0.22
        x1 = bx0 - gap
        x0 = x1 - bw
        mate = state.get("mate")
        ev = state.get("eval_white", 0.0)
        if mate is not None:
            frac = 1.0 if mate > 0 else 0.0
        else:
            frac = 1.0 / (1.0 + 2.718281828 ** (-ev / 350.0))   # logistic; eval is in centipawns
        frac = min(0.98, max(0.02, frac))
        white_bottom = state.get("white_bottom", True)
        c.create_rectangle(x0, by0, x1, by0 + board_h, fill=BAR_BLACK, outline="")
        if white_bottom:
            c.create_rectangle(x0, by0 + board_h * (1 - frac), x1, by0 + board_h,
                               fill=BAR_WHITE, outline="")
        else:
            c.create_rectangle(x0, by0, x1, by0 + board_h * frac, fill=BAR_WHITE, outline="")
        label = f"M{abs(mate)}" if mate is not None else f"{ev / 100:+.1f}"
        white_ahead = (mate if mate is not None else ev) >= 0
        near_white_end = (white_ahead == white_bottom)
        ty = (by0 + board_h - sq * 0.55) if near_white_end else (by0 + sq * 0.55)
        fill = BAR_BLACK if near_white_end else BAR_WHITE
        c.create_text((x0 + x1) / 2, ty, text=label, fill=fill,
                      font=("Consolas", max(8, int(sq * 0.2)), "bold"), angle=90)

    def _draw_arrow(self, c, arrow, sq):
        tk = self.tk
        x1, y1 = self._to_local(arrow[0], arrow[1])
        x2, y2 = self._to_local(arrow[2], arrow[3])
        width = max(5.0, sq * 0.17)
        head = (sq * 0.55, sq * 0.6, sq * 0.28)
        c.create_line(x1, y1, x2, y2, fill=ARROW_EDGE, width=width + 4,
                      arrow=tk.LAST, arrowshape=head, capstyle=tk.ROUND)
        c.create_line(x1, y1, x2, y2, fill=ARROW, width=width,
                      arrow=tk.LAST, arrowshape=head, capstyle=tk.ROUND)

    def _draw_panel(self, c, bx0, by1, board_w, state):
        lines = state.get("lines", [])
        status = state.get("status", "")
        if not lines and not status:
            return
        pad = 10
        fs = max(10, int(board_w * 0.028))
        line_h = fs + 7
        rows = ([status] if status else []) + list(lines)
        height = pad * 2 + line_h * len(rows)
        width = max(board_w, fs * 20)
        y0 = by1 + fs
        c.create_rectangle(bx0, y0, bx0 + width, y0 + height,
                           fill=PANEL_BG, outline="#2a3139", stipple="gray75")
        y = y0 + pad + fs * 0.6
        if status:
            c.create_text(bx0 + pad, y, text=status, anchor="w", fill=TEXT,
                          font=("Consolas", fs, "bold"))
            y += line_h
        for ln in lines:
            colour = MUTED if ln.startswith("  ") else TEXT
            c.create_text(bx0 + pad, y, text=ln, anchor="w", fill=colour,
                          font=("Consolas", fs))
            y += line_h


def _serve():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    try:
        win = _Window()
    except Exception:
        print("FAIL", flush=True)
        return
    print("READY", flush=True)
    win.serve()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        _serve()
