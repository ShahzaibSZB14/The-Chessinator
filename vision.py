"""Screen capture, board detection and piece recognition for the Chessinator.

Recognition is style-agnostic: instead of shipping piece images, we learn the
pieces from the starting position of the board on screen. Every square is
turned into a "signature" that is normalised against its own background
colour, so light/dark squares and last-move highlights don't matter much.
"""
import numpy as np
import cv2
import mss
import chess

SIG = 32                       # signature resolution (pixels per square side)
BOARD_PX = SIG * 8

# Circular weight mask: pieces live in the middle of the square, coordinate
# labels and highlight borders live in the corners/edges.
_yy, _xx = np.mgrid[0:SIG, 0:SIG] + 0.5
_ELLIPSE = ((_xx - SIG / 2) ** 2 + (_yy - SIG / 2) ** 2) <= (0.46 * SIG) ** 2
_BORDER = np.zeros((SIG, SIG), bool)
_BORDER[:2, :] = _BORDER[-2:, :] = _BORDER[:, :2] = _BORDER[:, -2:] = True

EMPTY = 0
NUM_CLASSES = 13


def piece_class(piece):
    if piece is None:
        return EMPTY
    return piece.piece_type + (0 if piece.color == chess.WHITE else 6)


class Screen:
    """Thin wrapper around mss using absolute virtual-desktop coordinates."""

    def __init__(self):
        self.sct = mss.mss()
        self.virtual = self.sct.monitors[0]

    def grab_all(self):
        """Whole virtual desktop as BGR image + its (left, top) offset."""
        m = self.virtual
        img = np.array(self.sct.grab(m))[:, :, :3]
        return img, (m["left"], m["top"])

    def grab_board(self, rect):
        """Board region resized to BOARD_PX x BOARD_PX (BGR, float32)."""
        x, y, w, h = rect
        region = {"left": int(round(x)), "top": int(round(y)),
                  "width": max(8, int(round(w))), "height": max(8, int(round(h)))}
        img = np.array(self.sct.grab(region))[:, :, :3]
        img = cv2.resize(img, (BOARD_PX, BOARD_PX), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)


# --------------------------------------------------------------------------
# Board detection
# --------------------------------------------------------------------------

def _checker_search(gray):
    """Coarse search for an 8x8 light/dark pattern. Returns (x, y, s) or None."""
    H, W = gray.shape
    best = (0.0, None)
    smin, smax = 48, min(H, W)
    s = smin
    while s <= smax:
        c = s / 8.0
        k = max(1, int(round(c * 0.8)))
        M = cv2.blur(gray, (k, k))           # mean of a cell-sized block, centred
        ny, nx = H - int(s), W - int(s)
        if ny <= 0 or nx <= 0:
            break
        step = max(1, int(c / 4))
        ys = np.arange(0, ny, step)
        xs = np.arange(0, nx, step)
        acc = np.zeros((len(ys), len(xs)), np.float32)
        absacc = np.zeros_like(acc)
        for i in range(8):
            for j in range(8):
                cy = (ys + int((i + 0.5) * c))[:, None]
                cx = (xs + int((j + 0.5) * c))[None, :]
                v = M[cy, cx]
                sign = 1.0 if (i + j) % 2 == 0 else -1.0   # top-left square is light
                acc += sign * v
                absacc += v
        score = acc / 64.0
        # mild preference for bigger boards (small thumbnails elsewhere on screen)
        score = score * (1.0 + 0.15 * np.log(s / smin + 1))
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        if score[iy, ix] > best[0]:
            best = (float(score[iy, ix]), (int(xs[ix]), int(ys[iy]), float(s)))
        s = int(s * 1.03) + 1
    return best[1] if best[0] > 6.0 else None


def _fit_lines(profile, start, size, lo, hi):
    """Best (start, size) so that the 7 inner grid lines hit gradient peaks."""
    idx = np.arange(len(profile), dtype=np.float32)
    starts = np.arange(start - lo, start + lo + 0.01, 0.5)
    sizes = np.arange(size * (1 - hi), size * (1 + hi) + 0.01, 0.5)
    k = np.arange(1, 8)
    S0, SZ, K = np.meshgrid(starts, sizes, k, indexing="ij")
    pos = S0 + K * SZ / 8.0 - 0.5           # gradient peak sits between pixels
    vals = np.interp(pos, idx, profile, left=0, right=0).sum(axis=2)
    a, b = np.unravel_index(np.argmax(vals), vals.shape)
    return float(starts[a]), float(sizes[b])


def refine_rect(gray, rect, search=0.06):
    """Pixel-accurate refinement using the grid lines between squares."""
    x, y, w, h = rect
    H, W = gray.shape
    pad = int(max(w, h) * search) + 4
    x0, y0 = max(0, int(x) - pad), max(0, int(y) - pad)
    x1, y1 = min(W, int(x + w) + pad), min(H, int(y + h) + pad)
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    if crop.shape[0] < 32 or crop.shape[1] < 32:
        return rect
    gx = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3))
    ry = slice(max(0, int(y) - y0), max(1, int(y + h) - y0))
    rx = slice(max(0, int(x) - x0), max(1, int(x + w) - x0))
    px = gx[ry, :].sum(axis=0)
    py = gy[:, rx].sum(axis=1)
    nx, nw = _fit_lines(px, x - x0, w, pad, search)
    ny, nh = _fit_lines(py, y - y0, h, pad, search)
    return (nx + x0, ny + y0, nw, nh)


def detect_board(img):
    """Find the chessboard in a BGR screenshot. Returns (x, y, w, h) or None."""
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 4 if max(gray_full.shape) > 1600 else 2
    small = cv2.resize(gray_full, None, fx=1 / scale, fy=1 / scale,
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    found = _checker_search(small)
    if found is None:
        return None
    sx, sy, ss = found
    rect = (sx * scale, sy * scale, ss * scale, ss * scale)
    rect = refine_rect(gray_full, rect, search=0.08)
    rect = _fix_cell_shift(gray_full, rect)
    rect = refine_rect(gray_full, rect, search=0.02)
    return rect


def checker_score(gray, rect):
    """Signed light/dark alternation of the 8x8 cells (top-left square light)."""
    x, y, w, h = rect
    H, W = gray.shape
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return -1e9
    crop = gray[int(y):int(y + h), int(x):int(x + w)].astype(np.float32)
    cells = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA)
    sign = np.where((np.add.outer(np.arange(8), np.arange(8)) % 2) == 0, 1.0, -1.0)
    return float((cells * sign).sum())


def _fix_cell_shift(gray, rect):
    """Grid-line fitting can lock on one square off; pick the best-aligned shift."""
    x, y, w, h = rect
    opts = [(x + i * w / 8, y + j * h / 8, w, h) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    return max(opts, key=lambda r: checker_score(gray, r))


# --------------------------------------------------------------------------
# Square signatures
# --------------------------------------------------------------------------

def signatures(board_img):
    """board_img: BOARD_PX^2 BGR float. Returns (fg, gray), each (64, P).

    Squares are indexed by screen position: idx = row*8 + col, row 0 = top.
    fg   = how strongly each pixel differs from the square's background colour
    gray = brightness of each pixel (tells white pieces from black ones)
    """
    sq = board_img.reshape(8, SIG, 8, SIG, 3).transpose(0, 2, 1, 3, 4).reshape(64, SIG, SIG, 3)
    bg = np.median(sq[:, _BORDER, :], axis=1)                     # (64, 3)
    diff = np.linalg.norm(sq - bg[:, None, None, :], axis=3)      # (64, SIG, SIG)
    fg = np.clip((diff - 14.0) / 40.0, 0.0, 1.0)
    gray = sq.mean(axis=3) / 255.0
    return fg[:, _ELLIPSE].astype(np.float32), gray[:, _ELLIPSE].astype(np.float32)


def sig_distance(fa, ga, fb, gb):
    """Pairwise distances between signature sets a (n) and b (m) -> (n, m)."""
    out = np.empty((len(fa), len(fb)), np.float32)
    for i in range(len(fa)):                     # loop keeps memory small
        mask = np.abs(fa[i][None] - fb).mean(axis=1)
        union = np.maximum(fa[i][None], fb)
        color = (union * np.abs(ga[i][None] - gb)).mean(axis=1)
        out[i] = mask + 1.5 * color
    return out


def frame_changed(a, b, tol=3.0):
    """True if two board images differ noticeably on any square."""
    d = np.abs(a - b).mean(axis=2)
    d = d.reshape(8, SIG, 8, SIG).mean(axis=(1, 3))
    return float(d.max()) > tol


class Recognizer:
    """Learns piece appearance from a known position and scores candidates."""

    MAX_EXTRA = 24     # adaptive prototypes kept per class

    def __init__(self, white_bottom=True):
        self.white_bottom = white_bottom
        self.base_fg = self.base_gray = self.base_cls = None
        self.extra = {c: [] for c in range(NUM_CLASSES)}

    # mapping between chess squares and screen indices ------------------
    def screen_index(self, square):
        f, r = chess.square_file(square), chess.square_rank(square)
        if self.white_bottom:
            return (7 - r) * 8 + f
        return r * 8 + (7 - f)

    def screen_classes(self, board):
        """Class of every screen square for a python-chess board."""
        cls = np.zeros(64, np.int64)
        for sq in chess.SQUARES:
            cls[self.screen_index(sq)] = piece_class(board.piece_at(sq))
        return cls

    # learning ---------------------------------------------------------
    def learn(self, fg, gray, board):
        self.base_fg, self.base_gray = fg.copy(), gray.copy()
        self.base_cls = self.screen_classes(board)
        self.extra = {c: [] for c in range(NUM_CLASSES)}

    def adapt(self, fg, gray, board, screen_idxs):
        """Remember how some squares look now (e.g. with move highlights)."""
        cls = self.screen_classes(board)
        for i in screen_idxs:
            lst = self.extra[int(cls[i])]
            lst.append((fg[i].copy(), gray[i].copy()))
            if len(lst) > self.MAX_EXTRA:
                lst.pop(0)

    def _protos(self):
        fgs, grs, cls = [self.base_fg], [self.base_gray], [self.base_cls]
        for c, lst in self.extra.items():
            if lst:
                fgs.append(np.stack([a for a, _ in lst]))
                grs.append(np.stack([b for _, b in lst]))
                cls.append(np.full(len(lst), c))
        return np.concatenate(fgs), np.concatenate(grs), np.concatenate(cls)

    def class_distances(self, fg, gray):
        """D[i, c] = distance of screen square i to the nearest class-c prototype."""
        pf, pg, pc = self._protos()
        d = sig_distance(fg, gray, pf, pg)
        D = np.full((64, NUM_CLASSES), 9.0, np.float32)
        for c in range(NUM_CLASSES):
            m = pc == c
            if m.any():
                D[:, c] = d[:, m].min(axis=1)
        return D

    def cost(self, D, board):
        cls = self.screen_classes(board)
        return float(D[np.arange(64), cls].sum()), cls

    # persistence ------------------------------------------------------
    def state(self):
        return {"white_bottom": self.white_bottom, "fg": self.base_fg,
                "gray": self.base_gray, "cls": self.base_cls}

    def load_state(self, st):
        self.white_bottom = st["white_bottom"]
        self.base_fg, self.base_gray, self.base_cls = st["fg"], st["gray"], st["cls"]
        self.extra = {c: [] for c in range(NUM_CLASSES)}


def white_is_bottom(fg, gray):
    """Bottom pieces brighter than top pieces -> White plays from the bottom."""
    def piece_brightness(rows):
        idx = [r * 8 + c for r in rows for c in range(8)]
        w = fg[idx]
        return float((w * gray[idx]).sum() / (w.sum() + 1e-6))
    return piece_brightness([6, 7]) > piece_brightness([0, 1])


def learn_start_position(fg, gray, white_bottom, allow_first_move):
    """Pick the start position that best explains the screen and learn from it.

    If we play Black, White may already have moved by the time we calibrate,
    so we also try every legal first move and keep the most self-consistent one.
    """
    rec = Recognizer(white_bottom)
    d = sig_distance(fg, gray, fg, gray)
    start = chess.Board()
    cands = [start]
    if allow_first_move:
        for mv in start.legal_moves:
            b = start.copy()
            b.push(mv)
            cands.append(b)

    def spread(board):
        cls = rec.screen_classes(board)
        total = 0.0
        for c in range(NUM_CLASSES):
            idx = np.where(cls == c)[0]
            if len(idx) > 1:
                total += d[np.ix_(idx, idx)].sum() / len(idx)
        return total

    best = min(cands, key=spread)
    rec.learn(fg, gray, best)
    return rec, best
