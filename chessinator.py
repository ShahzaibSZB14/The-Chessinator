"""The Chessinator - plays chess on any on-screen board using Stockfish.

Usage:  python chessinator.py [--level 8] [--profile NAME] [--color white|black]
Hotkeys while running:  F8 pause/resume   F10 new game   F9 quit
"""
import argparse
import ctypes
import os
import pickle
import shutil
import sys
import threading
import time

import chess
import chess.engine
import numpy as np
from pynput import keyboard, mouse

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, "profiles")

# Make screen capture and mouse use real pixels on scaled (125%/150%) displays.
# Must happen before mss / OpenCV create anything.
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import vision        # noqa: E402
import calibrate     # noqa: E402

# Lichess-style AI levels: (Skill Level, max depth, think time in seconds).
# Level 8 = full-strength Stockfish.
LEVELS = {
    1: (0, 5, 0.05), 2: (0, 5, 0.10), 3: (0, 5, 0.15), 4: (3, 5, 0.20),
    5: (7, 5, 0.30), 6: (11, 8, 0.40), 7: (16, 13, 0.50), 8: (20, 22, 1.00),
}

MARGIN = 0.12            # how much better a move must explain the screen than "nothing changed"
POLL = 0.08


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def ascii_board(board, white_bottom):
    ranks = range(7, -1, -1) if white_bottom else range(8)
    files = range(8) if white_bottom else range(7, -1, -1)
    lines = []
    for r in ranks:
        row = " ".join((board.piece_at(chess.square(f, r)) or ".").__str__() for f in files)
        lines.append(f"    {r + 1}  {row}")
    lines.append("       " + " ".join("abcdefgh"[f] for f in files))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Hotkeys
# --------------------------------------------------------------------------

class Controls:
    def __init__(self):
        self.paused = False
        self.quit = False
        self.new_game = False
        self.listener = keyboard.Listener(on_press=self._on_press, daemon=True)
        self.listener.start()

    def _on_press(self, key):
        if key == keyboard.Key.f8:
            self.paused = not self.paused
            log("PAUSED (F8 to resume)" if self.paused else "Resumed")
        elif key == keyboard.Key.f9:
            self.quit = True
            log("Quitting...")
        elif key == keyboard.Key.f10:
            self.new_game = True
            log("New game requested")

    def interrupted(self):
        return self.quit or self.new_game


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

def find_engine(path):
    cands = [path] if path else []
    cands += [os.path.join(HERE, "engine", "stockfish.exe"),
              os.path.join(HERE, "engine", "stockfish"),
              shutil.which("stockfish")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    sys.exit("Stockfish not found. Run:  python setup_stockfish.py   (or pass --engine PATH)")


class Engine:
    def __init__(self, path, level, threads, hash_mb):
        self.skill, self.depth, self.time = LEVELS[level]
        self.eng = chess.engine.SimpleEngine.popen_uci(path)
        opts = {"Threads": threads, "Hash": hash_mb, "Skill Level": self.skill}
        self.eng.configure({k: v for k, v in opts.items() if k in self.eng.options})

    def best_move(self, board):
        res = self.eng.play(board, chess.engine.Limit(depth=self.depth, time=self.time))
        return res.move

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------

class Chessinator:
    def __init__(self, args):
        self.args = args
        self.screen = vision.Screen()
        self.mouse = mouse.Controller()
        self.ctl = Controls()
        self.profile_path = os.path.join(PROFILES, args.profile + ".pkl")
        self.profile = self._load_profile()
        self.rect = None
        self.rec = None
        self.board = None
        self.my_color = None
        self.engine = Engine(find_engine(args.engine), args.level, args.threads, args.hash)

    # ---- profile -------------------------------------------------------
    def _load_profile(self):
        if os.path.isfile(self.profile_path):
            with open(self.profile_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_profile(self):
        os.makedirs(PROFILES, exist_ok=True)
        self.profile["rect"] = self.rect
        if self.rec is not None:
            self.profile["recognizer"] = self.rec.state()
        with open(self.profile_path, "wb") as f:
            pickle.dump(self.profile, f)

    # ---- geometry ------------------------------------------------------
    def square_center(self, square):
        idx = self.rec.screen_index(square)
        row, col = divmod(idx, 8)
        x, y, w, h = self.rect
        return (x + (col + 0.5) * w / 8, y + (row + 0.5) * h / 8)

    def off_board_point(self):
        x, y, w, h = self.rect
        return (x + w + w * 0.06, y + h / 2)

    # ---- vision helpers --------------------------------------------------
    def stable_frame(self, timeout=None):
        """Wait until the board stops changing (animations, drags) and return it."""
        t0 = time.time()
        prev = self.screen.grab_board(self.rect)
        while True:
            time.sleep(POLL)
            cur = self.screen.grab_board(self.rect)
            if not vision.frame_changed(prev, cur):
                return cur
            prev = cur
            if timeout and time.time() - t0 > timeout:
                return cur

    def observe(self):
        img = self.stable_frame(timeout=3.0)
        fg, gray = vision.signatures(img)
        return fg, gray, self.rec.class_distances(fg, gray)

    def best_candidate(self, D, cands):
        """cands: list of (label, board). Returns (label, board, cost, unexplained)."""
        scored = []
        for label, b in cands:
            c, cls = self.rec.cost(D, b)
            scored.append((c, label, b, cls))
        scored.sort(key=lambda s: s[0])
        c, label, b, cls = scored[0]
        # squares where some other piece/emptiness fits clearly better
        excess = D[np.arange(64), cls] - D.min(axis=1)
        unexplained = int((excess > 0.10).sum())
        return label, b, c, unexplained, scored

    # ---- setup ---------------------------------------------------------
    def setup_board_rect(self):
        rect = self.profile.get("rect")
        if rect and self.args.skip_calibration:
            self.rect = rect
            return
        log("Opening calibration window - line the grid up with the board, then press ENTER.")
        rect = calibrate.calibrate(self.screen, rect)
        if rect is None:
            sys.exit("Calibration cancelled.")
        self.rect = rect
        log(f"Board at x={rect[0]:.0f} y={rect[1]:.0f}, size {rect[2]:.1f} x {rect[3]:.1f} px")

    def setup_game(self):
        """Learn the pieces and figure out which side we play."""
        if self.args.fen:
            st = self.profile.get("recognizer")
            if not st:
                sys.exit("--fen needs a profile that was calibrated on a starting position first.")
            self.rec = vision.Recognizer()
            self.rec.load_state(st)
            if self.args.color:
                self.rec.white_bottom = self.args.color == "white"
            self.board = chess.Board(self.args.fen)
            self.args.fen = None         # only for the first game
        else:
            input("\n>>> Show the STARTING POSITION on screen (as Black, White's first move is OK)"
                  "\n    and press ENTER here... ")
            img = self.stable_frame(timeout=3.0)
            fg, gray = vision.signatures(img)
            if self.args.color:
                white_bottom = self.args.color == "white"
            else:
                white_bottom = vision.white_is_bottom(fg, gray)
            self.rec, self.board = vision.learn_start_position(
                fg, gray, white_bottom, allow_first_move=not white_bottom)
            self._self_check(fg, gray)

        self.my_color = chess.WHITE if self.rec.white_bottom else chess.BLACK
        print()
        print(ascii_board(self.board, self.rec.white_bottom))
        print(f"\n    Playing as {'WHITE' if self.my_color else 'BLACK'} (bottom side).")
        ans = input("    Correct?  [ENTER = yes / f = flip colours / q = quit] ").strip().lower()
        if ans == "q":
            sys.exit(0)
        if ans == "f":
            self.args.color = "black" if self.my_color == chess.WHITE else "white"
            return self.setup_game()
        self._save_profile()

    def _self_check(self, fg, gray):
        """Warn if pieces are hard to tell apart with this board/piece theme."""
        D = vision.sig_distance(fg, gray, fg, gray)
        cls = self.rec.base_cls
        bad = 0
        for i in range(64):
            same = (cls == cls[i]) & (np.arange(64) != i)
            if not same.any():
                continue
            other = cls != cls[i]
            if D[i, same].min() >= D[i, other].min():
                bad += 1
        if bad:
            log(f"Warning: {bad} squares look ambiguous - recognition may be unreliable. "
                "Re-check the calibration grid.")
        else:
            log("Pieces learned - all squares are clearly distinguishable.")

    # ---- acting ----------------------------------------------------------
    def _wait(self, sec):
        time.sleep(sec)

    def play_on_screen(self, move):
        user_pos = self.mouse.position
        a = self.square_center(move.from_square)
        b = self.square_center(move.to_square)
        if self.args.input == "click":
            self._click(a)
            self._wait(0.06)
            self._click(b)
        else:
            self.mouse.position = a
            self._wait(0.03)
            self.mouse.press(mouse.Button.left)
            steps = 10
            for i in range(1, steps + 1):
                t = i / steps
                self.mouse.position = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                self._wait(0.012)
            self._wait(0.03)
            self.mouse.release(mouse.Button.left)
        if move.promotion:
            self._wait(0.35)
            # promotion pickers list Q, N, R, B starting on the target square, toward the centre
            order = [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]
            k = order.index(move.promotion)
            x, y, w, h = self.rect
            row = self.rec.screen_index(move.to_square) // 8
            direction = 1 if row == 0 else -1
            self._click((b[0], b[1] + direction * k * h / 8))
        self._wait(0.05)
        # put the mouse back where the user had it (or beside the board)
        ux, uy = user_pos
        x, y, w, h = self.rect
        on_board = x <= ux <= x + w and y <= uy <= y + h
        self.mouse.position = self.off_board_point() if on_board else user_pos

    def _click(self, p):
        self.mouse.position = p
        self._wait(0.03)
        self.mouse.click(mouse.Button.left)

    # ---- main loop -------------------------------------------------------
    def wait_until_resumed(self):
        while self.ctl.paused and not self.ctl.interrupted():
            time.sleep(0.1)

    def our_turn(self):
        move = self.engine.best_move(self.board)
        log(f"Stockfish plays {self.board.san(move)}")
        before = self.board.copy()
        after = self.board.copy()
        after.push(move)
        for attempt in range(3):
            self.wait_until_resumed()
            if self.ctl.interrupted():
                return
            self.play_on_screen(move)
            t0 = time.time()
            while time.time() - t0 < 3.0 and not self.ctl.interrupted():
                fg, gray, D = self.observe()
                cands = [("none", before), ("mine", after)]
                for reply in after.legal_moves:      # opponent may reply instantly
                    b = after.copy()
                    b.push(reply)
                    cands.append((reply, b))
                label, b, cost, unexplained, scored = self.best_candidate(D, cands)
                none_cost = next(s[0] for s in scored if s[1] == "none")
                if label == "none" or none_cost - cost < MARGIN:
                    time.sleep(0.15)
                    continue
                self.board = after
                self._adapt(fg, gray, before, move)
                if label != "mine":
                    self._accept_opponent(label, fg, gray)
                return
            if self.ctl.interrupted():
                return
            log("Move didn't show up on the board - trying again...")
        log("Could not play the move. Paused - make the move by hand, then press F8.")
        self.board = after
        self.ctl.paused = True

    def _adapt(self, fg, gray, before, move):
        """Learn how the squares of a confirmed move look (highlights etc.)."""
        b = before.copy()
        b.push(move)
        squares = {move.from_square, move.to_square}
        if before.is_castling(move):
            rank = chess.square_rank(move.from_square)
            squares |= {chess.square(f, rank) for f in (0, 3, 5, 7)}
        if before.is_en_passant(move):
            squares.add(chess.square(chess.square_file(move.to_square),
                                     chess.square_rank(move.from_square)))
        self.rec.adapt(fg, gray, b, [self.rec.screen_index(s) for s in squares])

    def _accept_opponent(self, move, fg, gray):
        before = self.board.copy()
        log(f"Opponent played {before.san(move)}")
        self.board.push(move)
        self._adapt(fg, gray, before, move)

    def their_turn(self):
        confirm = None
        mismatch_since = None
        while not self.ctl.interrupted():
            self.wait_until_resumed()
            if self.ctl.interrupted():
                return
            fg, gray, D = self.observe()
            cands = [(None, self.board)]
            for mv in self.board.legal_moves:
                b = self.board.copy()
                b.push(mv)
                cands.append((mv, b))
            label, b, cost, unexplained, scored = self.best_candidate(D, cands)
            none_cost = next(s[0] for s in scored if s[1] is None)

            if unexplained >= 3:
                mismatch_since = mismatch_since or time.time()
                if time.time() - mismatch_since > 6:
                    log("The screen doesn't match the game I'm tracking (board moved? new game?). "
                        "F8 = pause, F10 = new game, or re-run calibration.")
                    mismatch_since = time.time() + 20
            else:
                mismatch_since = None

            if label is not None and none_cost - cost > MARGIN:
                if confirm == label:                 # seen twice in a row -> accept
                    self._accept_opponent(label, fg, gray)
                    return
                confirm = label
            else:
                confirm = None
            time.sleep(POLL)

    def play_game(self):
        self.ctl.new_game = False
        log("Playing! F8 = pause/resume, F10 = new game, F9 = quit")
        while not self.ctl.interrupted():
            self.wait_until_resumed()
            if self.board.is_game_over():
                log(f"Game over: {self.board.result()} ({self.board.outcome().termination.name})")
                return
            if self.board.turn == self.my_color:
                self.our_turn()
            else:
                self.their_turn()

    def run(self):
        try:
            self.setup_board_rect()
            while not self.ctl.quit:
                self.setup_game()
                self.play_game()
                if self.ctl.quit:
                    break
                if not self.ctl.new_game:
                    ans = input("\n>>> Play another game? [ENTER = yes / q = quit] ").strip().lower()
                    if ans == "q":
                        break
        finally:
            self.engine.close()


def main():
    p = argparse.ArgumentParser(description="The Chessinator - Stockfish plays on any on-screen board.")
    p.add_argument("--level", type=int, default=8, choices=range(1, 9),
                   help="Stockfish level 1-8 (8 = full strength, default)")
    p.add_argument("--profile", default="default",
                   help="name to save/load board calibration under (e.g. lichess, chesscom)")
    p.add_argument("--skip-calibration", action="store_true",
                   help="reuse the saved board position without opening the calibration window")
    p.add_argument("--color", choices=["white", "black"],
                   help="force the side we play (default: auto-detect, bottom side)")
    p.add_argument("--fen", help="start from this position (needs a profile with learned pieces)")
    p.add_argument("--input", choices=["drag", "click"], default="drag",
                   help="how to move pieces: drag-and-drop (default) or click-click")
    p.add_argument("--engine", help="path to a UCI engine (default: engine/stockfish.exe)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--hash", type=int, default=128, help="engine hash size in MB")
    args = p.parse_args()
    Chessinator(args).run()


if __name__ == "__main__":
    main()
