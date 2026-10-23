"""Interactive board calibration window (OpenCV).

Shows the auto-detected board with an 8x8 grid on top, plus four zoomed
corner views so the edges can be lined up to the exact pixel. Boards can be
a plain square, or any 4-cornered shape for tilted / perspective views.
"""
import numpy as np
import cv2

import vision

WIN = "Chessinator - board calibration"
MAX_VIEW_W, MAX_VIEW_H = 1000, 720
ZOOM_SRC, ZOOM = 36, 4          # corner magnifier: 36x36 source px shown at 4x
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")

KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440

HELP = [
    "Drag inside = move   Drag a corner = resize   P = square / perspective (free corners) mode",
    "Arrows = move 1px (just the selected corner if one is selected)   W/A/S/D = 10px",
    "1-4 = select corner, 0 = none   [ ] = shrink/grow   O = rotate grid 90 deg",
    "M = manual (click the corners)   R = auto-detect   Z = whole screen   ENTER = accept   ESC = cancel",
]


def _is_square(q):
    """True if the quad is (close to) an axis-aligned square."""
    w, h = q[1, 0] - q[0, 0], q[3, 1] - q[0, 1]
    ok = (abs(q[0, 1] - q[1, 1]) < 1 and abs(q[3, 1] - q[2, 1]) < 1 and
          abs(q[0, 0] - q[3, 0]) < 1 and abs(q[1, 0] - q[2, 0]) < 1)
    return ok and abs(w - h) <= max(2.0, 0.02 * w)


class Calibrator:
    def __init__(self, img, quad):
        self.img = img
        self.H, self.W = img.shape[:2]
        if quad is None:
            s = self.H / 2
            quad = vision.rect_to_quad((self.W / 2 - s / 2, self.H / 4, s, s))
            self.full_view = True
        else:
            self.full_view = False
        self.q = vision.as_quad(quad).copy()
        self.perspective = not _is_square(self.q)
        self.sel = None              # selected corner index
        self.drag = None
        self.manual_pts = None
        self._set_view()

    # ---- view transform ------------------------------------------------
    def _set_view(self):
        if self.full_view:
            vx0, vy0, vx1, vy1 = 0, 0, self.W, self.H
        else:
            (x0, y0), (x1, y1) = self.q.min(axis=0), self.q.max(axis=0)
            m = 0.18 * max(x1 - x0, y1 - y0)
            vx0, vy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
            vx1, vy1 = min(self.W, int(x1 + m)), min(self.H, int(y1 + m))
        self.vx0, self.vy0 = vx0, vy0
        self.scale = min(MAX_VIEW_W / (vx1 - vx0), MAX_VIEW_H / (vy1 - vy0), 2.0)
        self.view = cv2.resize(self.img[vy0:vy1, vx0:vx1], None, fx=self.scale,
                               fy=self.scale, interpolation=cv2.INTER_AREA)

    def to_view(self, x, y):
        return int(round((x - self.vx0) * self.scale)), int(round((y - self.vy0) * self.scale))

    def to_src(self, vx, vy):
        return vx / self.scale + self.vx0, vy / self.scale + self.vy0

    # ---- drawing -------------------------------------------------------
    def _draw_grid(self, canvas, tf):
        Hm = cv2.getPerspectiveTransform(vision._UNIT, self.q.astype(np.float32))

        def P(u, v):
            p = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), Hm)[0, 0]
            return tf(p[0], p[1])

        for k in range(9):
            col = (0, 255, 0) if k in (0, 8) else (255, 255, 0)
            cv2.line(canvas, P(k / 8, 0), P(k / 8, 1), col, 1, cv2.LINE_AA)
            cv2.line(canvas, P(0, k / 8), P(1, k / 8), col, 1, cv2.LINE_AA)

    def _zoom(self, cx, cy):
        s = ZOOM_SRC
        x0, y0 = int(cx) - s // 2, int(cy) - s // 2
        patch = np.zeros((s, s, 3), np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.W, x0 + s), min(self.H, y0 + s)
        if sx1 > sx0 and sy1 > sy0:
            patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.img[sy0:sy1, sx0:sx1]
        big = cv2.resize(patch, (s * ZOOM, s * ZOOM), interpolation=cv2.INTER_NEAREST)
        self._draw_grid(big, lambda px, py: (int(round((px - x0) * ZOOM)), int(round((py - y0) * ZOOM))))
        return big

    def render(self):
        canvas = self.view.copy()
        if self.manual_pts is None:
            self._draw_grid(canvas, self.to_view)
            for i, c in enumerate(self.q):
                col = (0, 165, 255) if i == self.sel else (0, 0, 255)
                cv2.circle(canvas, self.to_view(*c), 7 if i == self.sel else 6, col, 2)
            mode = "PERSPECTIVE (free corners)" if self.perspective else "SQUARE"
            w = np.linalg.norm(self.q[1] - self.q[0])
            cv2.putText(canvas, f"{mode}   top edge {w:.1f}px", (8, canvas.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            need = 4 if self.perspective else 2
            names = CORNER_NAMES if self.perspective else ("top-left", "bottom-right")
            msg = f"Click the {names[len(self.manual_pts)].upper()} corner of the board" \
                if len(self.manual_pts) < need else ""
            cv2.putText(canvas, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
            for p in self.manual_pts:
                cv2.circle(canvas, self.to_view(*p), 6, (0, 0, 255), 2)

        tiles = []
        for name, c in zip(CORNER_NAMES, self.q):
            z = self._zoom(*c)
            cv2.putText(z, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(z, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(z, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        zooms = np.hstack(tiles)
        width = max(canvas.shape[1], zooms.shape[1], 760)

        def pad(a):
            return cv2.copyMakeBorder(a, 0, 0, 0, width - a.shape[1], cv2.BORDER_CONSTANT)
        help_img = np.full((22 * len(HELP) + 8, width, 3), 30, np.uint8)
        for i, line in enumerate(HELP):
            cv2.putText(help_img, line, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (230, 230, 230), 1, cv2.LINE_AA)
        self.canvas_h = canvas.shape[0]
        return np.vstack([pad(canvas), pad(zooms), help_img])

    # ---- editing -------------------------------------------------------
    def _set_corner(self, i, x, y, base):
        if self.perspective:
            q = base.copy()
            q[i] = (x, y)
            self.q = q
            return
        # square mode: opposite corner stays fixed
        o = base[(i + 2) % 4]
        size = max(abs(x - o[0]), abs(y - o[1]), 16)
        sx = -1 if i in (0, 3) else 1
        sy = -1 if i in (0, 1) else 1
        a = (o[0] + sx * size, o[1] + sy * size)
        x0, x1 = sorted((o[0], a[0]))
        y0, y1 = sorted((o[1], a[1]))
        self.q = vision.rect_to_quad((x0, y0, x1 - x0, y1 - y0))

    def _grow(self, px):
        c = self.q.mean(axis=0)
        r = np.linalg.norm(self.q[0] - c)
        self.q = c + (self.q - c) * (r + px) / max(r, 1)

    def _nudge(self, dx, dy):
        if self.sel is not None and self.perspective:
            self.q[self.sel] += (dx, dy)
        else:
            self.q += (dx, dy)

    def on_mouse(self, event, vx, vy, flags, _):
        if vy >= self.canvas_h and self.drag is None:
            return
        sx, sy = self.to_src(vx, vy)
        if self.manual_pts is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.manual_pts.append((sx, sy))
                if self.perspective and len(self.manual_pts) == 4:
                    self.q = np.array(self.manual_pts, np.float64)
                elif not self.perspective and len(self.manual_pts) == 2:
                    (ax, ay), (bx, by) = self.manual_pts
                    self.q = vision.rect_to_quad((min(ax, bx), min(ay, by), abs(bx - ax), abs(by - ay)))
                else:
                    return
                self.manual_pts = None
                self.full_view = False
                self._set_view()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            grab = 12 / self.scale
            d = np.linalg.norm(self.q - (sx, sy), axis=1)
            if d.min() < grab:
                i = int(np.argmin(d))
                self.sel = i
                self.drag = ("corner", i, self.q.copy(), None)
            elif vision.quad_contains(self.q, sx, sy):
                self.drag = ("move", None, self.q.copy(), (sx, sy))
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            kind, i, q0, start = self.drag
            if kind == "move":
                self.q = q0 + (sx - start[0], sy - start[1])
            else:
                self._set_corner(i, sx, sy, q0)
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
            ch = chr(k).lower() if 0 <= k < 256 else ""
            if k in (13, 10):
                cv2.destroyWindow(WIN)
                return self.q.copy()
            if k == 27:
                cv2.destroyWindow(WIN)
                return None
            if k == KEY_LEFT: self._nudge(-1, 0)
            elif k == KEY_RIGHT: self._nudge(1, 0)
            elif k == KEY_UP: self._nudge(0, -1)
            elif k == KEY_DOWN: self._nudge(0, 1)
            elif ch == "a": self._nudge(-10, 0)
            elif ch == "d": self._nudge(10, 0)
            elif ch == "w": self._nudge(0, -10)
            elif ch == "s": self._nudge(0, 10)
            elif ch == "[": self._grow(-1)
            elif ch == "]": self._grow(1)
            elif ch in "1234" and ch:
                self.sel = int(ch) - 1
            elif ch == "0":
                self.sel = None
            elif ch == "o":
                self.q = np.roll(self.q, -1, axis=0)
            elif ch == "p":
                self.perspective = not self.perspective
                if not self.perspective:      # snap back to the bounding square
                    (x0, y0), (x1, y1) = self.q.min(axis=0), self.q.max(axis=0)
                    s = max(x1 - x0, y1 - y0)
                    self.q = vision.rect_to_quad((x0, y0, s, s))
            elif ch == "m":
                self.manual_pts = []
            elif ch == "z":
                self.full_view = not self.full_view
                self._set_view()
            elif ch == "r":
                found = vision.detect_board(self.img)
                if found is not None:
                    self.q = vision.as_quad(found).copy()
                    self.perspective = not _is_square(self.q)
                    self.full_view = False
                    self._set_view()
                else:
                    print("  Auto-detect found nothing - use M for manual.")


def calibrate(screen, quad=None):
    """Return the board's 4 corners in absolute screen coordinates, or None if cancelled."""
    img, (ox, oy) = screen.grab_all()
    local = None
    if quad is not None:
        local = vision.as_quad(quad) - (ox, oy)
        if _is_square(local):
            x, y = local[0]
            w, h = local[2] - local[0]
            r = vision.refine_rect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (x, y, w, h), 0.02)
            local = vision.rect_to_quad(r)
    else:
        print("  Auto-detecting the board...")
        local = vision.detect_board(img)
        if local is None:
            print("  Could not find a board automatically. Press M and click its corners "
                  "(press P first if the board is tilted).")
    res = Calibrator(img, local).run()
    if res is None:
        return None
    return res + (ox, oy)
