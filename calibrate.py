"""Interactive board calibration window (OpenCV).

Shows the auto-detected board with an 8x8 grid on top, plus four zoomed
corner views so the edges can be lined up to the exact pixel.
"""
import numpy as np
import cv2

import vision

WIN = "Chessinator - board calibration"
MAX_VIEW_W, MAX_VIEW_H = 1000, 720
ZOOM_SRC, ZOOM = 36, 4          # corner magnifier: 36x36 source px shown at 4x

KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440

HELP = [
    "Drag inside the box = move   |  Drag a corner = resize",
    "Arrows = move 1px   [ ] = size -/+ 1px   Shift-style big steps: A/D/W/S = 10px",
    "M = manual (click top-left then bottom-right corner)   R = auto-detect again",
    "Z = toggle whole-screen view   ENTER = accept   ESC = cancel",
]


class Calibrator:
    def __init__(self, img, rect):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.rect = list(rect) if rect else [self.W / 4, self.H / 4, self.H / 2, self.H / 2]
        self.full_view = rect is None
        self.drag = None
        self.manual_pts = None
        self._set_view()

    # ---- view transform ------------------------------------------------
    def _set_view(self):
        if self.full_view:
            vx0, vy0, vx1, vy1 = 0, 0, self.W, self.H
        else:
            x, y, w, h = self.rect
            m = 0.18 * max(w, h)
            vx0, vy0 = max(0, int(x - m)), max(0, int(y - m))
            vx1, vy1 = min(self.W, int(x + w + m)), min(self.H, int(y + h + m))
        self.vx0, self.vy0 = vx0, vy0
        self.scale = min(MAX_VIEW_W / (vx1 - vx0), MAX_VIEW_H / (vy1 - vy0), 2.0)
        self.view = cv2.resize(self.img[vy0:vy1, vx0:vx1], None, fx=self.scale,
                               fy=self.scale, interpolation=cv2.INTER_AREA)

    def to_view(self, x, y):
        return int(round((x - self.vx0) * self.scale)), int(round((y - self.vy0) * self.scale))

    def to_src(self, vx, vy):
        return vx / self.scale + self.vx0, vy / self.scale + self.vy0

    def corners(self):
        x, y, w, h = self.rect
        return [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

    # ---- drawing -------------------------------------------------------
    def _draw_grid(self, canvas, tf, thick=1):
        x, y, w, h = self.rect
        for k in range(9):
            col = (0, 255, 0) if k in (0, 8) else (255, 255, 0)
            a = tf(x + k * w / 8, y)
            b = tf(x + k * w / 8, y + h)
            cv2.line(canvas, a, b, col, thick, cv2.LINE_AA)
            a = tf(x, y + k * h / 8)
            b = tf(x + w, y + k * h / 8)
            cv2.line(canvas, a, b, col, thick, cv2.LINE_AA)

    def _zoom(self, cx, cy):
        s = ZOOM_SRC
        x0, y0 = int(cx) - s // 2, int(cy) - s // 2
        patch = np.zeros((s, s, 3), np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.W, x0 + s), min(self.H, y0 + s)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.img[sy0:sy1, sx0:sx1]
        big = cv2.resize(patch, (s * ZOOM, s * ZOOM), interpolation=cv2.INTER_NEAREST)
        tf = lambda px, py: (int(round((px - x0) * ZOOM)), int(round((py - y0) * ZOOM)))
        self._draw_grid(big, tf, 1)
        return big

    def render(self):
        canvas = self.view.copy()
        if self.manual_pts is None:
            self._draw_grid(canvas, self.to_view, 1)
            for c in self.corners():
                cv2.circle(canvas, self.to_view(*c), 6, (0, 0, 255), 2)
            x, y, w, h = self.rect
            cv2.putText(canvas, f"{w:.1f} x {h:.1f} px  @ ({x:.0f},{y:.0f})",
                        (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
        else:
            msg = "Click the TOP-LEFT corner of the board" if not self.manual_pts \
                else "Now click the BOTTOM-RIGHT corner"
            cv2.putText(canvas, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
            for p in self.manual_pts:
                cv2.circle(canvas, self.to_view(*p), 6, (0, 0, 255), 2)

        tiles = []
        for name, c in zip(("top-left", "top-right", "bottom-left", "bottom-right"), self.corners()):
            z = self._zoom(*c)
            cv2.putText(z, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(z, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(z, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        zooms = np.hstack(tiles)
        width = max(canvas.shape[1], zooms.shape[1])
        def pad(a):
            return cv2.copyMakeBorder(a, 0, 0, 0, width - a.shape[1], cv2.BORDER_CONSTANT)
        help_img = np.full((22 * len(HELP) + 8, width, 3), 30, np.uint8)
        for i, line in enumerate(HELP):
            cv2.putText(help_img, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (230, 230, 230), 1, cv2.LINE_AA)
        self.canvas_h = canvas.shape[0]
        return np.vstack([pad(canvas), pad(zooms), help_img])

    # ---- interaction ---------------------------------------------------
    def on_mouse(self, event, vx, vy, flags, _):
        if vy >= self.canvas_h and self.drag is None:
            return
        sx, sy = self.to_src(vx, vy)
        if self.manual_pts is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.manual_pts.append((sx, sy))
                if len(self.manual_pts) == 2:
                    (ax, ay), (bx, by) = self.manual_pts
                    self.rect = [min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)]
                    self.manual_pts = None
                    self.full_view = False
                    self._set_view()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            grab = 12 / self.scale
            for i, (cx, cy) in enumerate(self.corners()):
                if abs(sx - cx) < grab and abs(sy - cy) < grab:
                    self.drag = ("corner", i, list(self.rect))
                    break
            else:
                x, y, w, h = self.rect
                if x <= sx <= x + w and y <= sy <= y + h:
                    self.drag = ("move", (sx, sy), list(self.rect))
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            kind, info, r0 = self.drag
            x, y, w, h = r0
            if kind == "move":
                self.rect = [x + sx - info[0], y + sy - info[1], w, h]
            else:
                # opposite corner stays fixed; keep the board square
                ox = x + w if info in (0, 2) else x
                oy = y + h if info in (0, 1) else y
                size = max(abs(sx - ox), abs(sy - oy), 16)
                nx = ox - size if info in (0, 2) else ox
                ny = oy - size if info in (0, 1) else oy
                self.rect = [nx, ny, size, size]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKeyEx(20)
            if k == -1:
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    return None
                continue
            x, y, w, h = self.rect
            ch = chr(k).lower() if 0 <= k < 256 else ""
            if k in (13, 10):
                cv2.destroyWindow(WIN)
                return tuple(self.rect)
            if k == 27:
                cv2.destroyWindow(WIN)
                return None
            if k == KEY_LEFT: self.rect[0] -= 1
            elif k == KEY_RIGHT: self.rect[0] += 1
            elif k == KEY_UP: self.rect[1] -= 1
            elif k == KEY_DOWN: self.rect[1] += 1
            elif ch == "a": self.rect[0] -= 10
            elif ch == "d": self.rect[0] += 10
            elif ch == "w": self.rect[1] -= 10
            elif ch == "s": self.rect[1] += 10
            elif ch == "[": self.rect[2] -= 1; self.rect[3] -= 1
            elif ch == "]": self.rect[2] += 1; self.rect[3] += 1
            elif ch == "m":
                self.manual_pts = []
            elif ch == "z":
                self.full_view = not self.full_view
                self._set_view()
            elif ch == "r":
                found = vision.detect_board(self.img)
                if found:
                    self.rect = list(found)
                    self.full_view = False
                    self._set_view()
                else:
                    print("  Auto-detect found nothing - use M for manual.")


def calibrate(screen, rect=None):
    """Return board rect in absolute screen coordinates, or None if cancelled."""
    img, (ox, oy) = screen.grab_all()
    local = None
    if rect is not None:
        local = (rect[0] - ox, rect[1] - oy, rect[2], rect[3])
        local = vision.refine_rect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), local, 0.02)
    else:
        print("  Auto-detecting the board...")
        local = vision.detect_board(img)
        if local is None:
            print("  Could not find a board automatically. Press M and click its corners.")
    res = Calibrator(img, local).run()
    if res is None:
        return None
    return (res[0] + ox, res[1] + oy, res[2], res[3])
