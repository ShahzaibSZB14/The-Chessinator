"""The Chessinator - plays chess on any on-screen board using Stockfish.

Usage:  python chessinator.py [--level 8] [--profile NAME] [--color white|black]
Hotkeys while running (defaults):  F8 pause/resume   F10 new game   F9 quit
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
from pynput import keyboard

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
import console       # noqa: E402
import mouse_input   # noqa: E402
import timecontrol   # noqa: E402

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

def parse_key(name):
    name = name.strip().lower()
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise SystemExit(f"Unknown hotkey: {name}")


class Controls:
    def __init__(self, keys="f8,f10,f9"):
        names = [k.strip() for k in keys.split(",")]
        if len(names) != 3:
            raise SystemExit("--keys needs three keys: pause,newgame,quit")
        self.names = [n.upper() for n in names]
        self.k_pause, self.k_new, self.k_quit = (parse_key(n) for n in names)
        self.paused = False
        self.quit = False
        self.new_game = False
        self.listener = keyboard.Listener(on_press=self._on_press, daemon=True)
        self.listener.start()

    def help(self):
        p, n, q = self.names
        return f"{p} = pause/resume, {n} = new game, {q} = quit"

    def _on_press(self, key):
        if key == self.k_pause:
            self.paused = not self.paused
            log(f"PAUSED ({self.names[0]} to resume)" if self.paused else "Resumed")
        elif key == self.k_quit:
            self.quit = True
            log("Quitting...")
        elif key == self.k_new:
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

    def best_move(self, board, clock=None, lag=0.0):
        """Fixed per-level think time, or a per-move budget from the clock when a
        time control is set (the level's depth cap and skill still apply)."""
        budget = self.time
        if clock is not None:
            budget = min(self.time if self.skill < 20 else 60.0, clock.budget(board, lag))
        return self.eng.play(board, chess.engine.Limit(depth=self.depth, time=budget)).move

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
        self.hand = mouse_input.Hand(mouse_input.make_mouse(args.mouse), settle=args.settle,
                                     hold=args.hold, drag_time=args.drag_time,
                                     click_gap=args.click_gap)
        self.method = "drag" if args.input in ("auto", "drag") else "click"
        self.ctl = Controls(args.keys)
        self.profile_path = os.path.join(PROFILES, args.profile + ".pkl")
        self.profile = self._load_profile()
        self.quad = None
        self.rec = None
        self.board = None
        self.my_color = None
        self.clock = None
        self.engine = Engine(find_engine(args.engine), args.level, args.threads, args.hash)

    # ---- profile -------------------------------------------------------
    def _load_profile(self):
        if os.path.isfile(self.profile_path):
            with open(self.profile_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_profile(self):
        os.makedirs(PROFILES, exist_ok=True)
        self.profile["quad"] = self.quad
        if self.rec is not None:
            self.profile["recognizer"] = self.rec.state()
        with open(self.profile_path, "wb") as f:
            pickle.dump(self.profile, f)

    # ---- geometry ------------------------------------------------------
    def screen_point(self, row, col):
        """Centre of a screen-grid cell (row 0 = top as seen) in screen pixels."""
        return vision.board_to_screen(self.quad, (col + 0.5) / 8, (row + 0.5) / 8)

    def square_center(self, square):
        return self.screen_point(*divmod(self.rec.screen_index(square), 8))

    def off_board_point(self):
        q = vision.as_quad(self.quad)
        width = q[:, 0].max() - q[:, 0].min()
        return (q[:, 0].max() + width * 0.06, q[:, 1].mean())

    # ---- vision helpers --------------------------------------------------
    def stable_frame(self, timeout=None):
        """Wait until the board stops changing (animations, drags) and return it."""
        t0 = time.time()
        prev = self.screen.grab_board(self.quad)
        while True:
            time.sleep(POLL)
            cur = self.screen.grab_board(self.quad)
            if not vision.frame_changed(prev, cur, self.args.stable_tol):
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
        quad = self.profile.get("quad", self.profile.get("rect"))
        if quad is not None:
            quad = vision.as_quad(quad)
        if quad is not None and self.args.skip_calibration:
            self.quad = quad
            return
        log("Opening calibration window - line the grid up with the board, then press ENTER.")
        with console.hidden():                # keep the console out of the screenshot
            quad = calibrate.calibrate(self.screen, quad)
        if quad is None:
            sys.exit("Calibration cancelled.")
        self.quad = quad
        log("Board corners: " + "  ".join(f"({x:.0f},{y:.0f})" for x, y in quad))

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
    def play_on_screen(self, move, method):
        user_pos = self.hand.position
        a = self.square_center(move.from_square)
        b = self.square_center(move.to_square)
        if method == "click":
            self.hand.click(a)
            time.sleep(self.hand.click_gap)
            self.hand.click(b)
        else:
            self.hand.drag(a, b)
        if move.promotion:
            time.sleep(0.35 + self.hand.click_gap)
            # promotion pickers list Q, N, R, B starting on the target square, toward the centre
            order = [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]
            k = order.index(move.promotion)
            row, col = divmod(self.rec.screen_index(move.to_square), 8)
            direction = 1 if row == 0 else -1
            self.hand.click(self.screen_point(row + direction * k, col))
        time.sleep(0.05)
        # put the mouse back where the user had it (or beside the board)
        on_board = vision.quad_contains(self.quad, *user_pos)
        self.hand.move_to(self.off_board_point() if on_board else user_pos)

    # ---- main loop -------------------------------------------------------
    def wait_until_resumed(self):
        while self.ctl.paused and not self.ctl.interrupted():
            time.sleep(0.1)

    def _clock_text(self):
        if not self.clock:
            return ""
        t = self.board.turn
        me = self.clock.remaining(self.my_color, t)
        opp = self.clock.remaining(not self.my_color, t)
        return f"   [clock  you {timecontrol.fmt(me)} | opp {timecontrol.fmt(opp)}]"

    def _default_lag(self):
        a = self.args
        return a.settle + a.hold + (a.drag_time if self.method == "drag" else a.click_gap + a.hold) + 0.35

    def our_turn(self):
        turn_start = self.clock.turn_started if self.clock else time.time()
        t0 = time.time()
        lag = self.clock.expected_lag(self._default_lag()) if self.clock else 0.0
        move = self.engine.best_move(self.board, self.clock, lag)
        think = time.time() - t0
        log(f"Stockfish plays {self.board.san(move)}  ({think:.1f}s){self._clock_text()}")
        before = self.board.copy()
        after = self.board.copy()
        after.push(move)
        for attempt in range(4):
            self.wait_until_resumed()
            if self.ctl.interrupted():
                return
            method = self.method
            if self.args.input == "auto" and attempt % 2 == 1:
                method = "click" if self.method == "drag" else "drag"
            self.play_on_screen(move, method)
            move_done = t0 = time.time()
            while time.time() - t0 < self.args.move_timeout and not self.ctl.interrupted():
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
                if method != self.method:
                    log(f"'{method}' input works here - using it from now on.")
                    self.method = method
                self.board = after
                self._adapt(fg, gray, before, move)
                if self.clock:
                    self.clock.end_turn(self.my_color, move_done)
                    self.clock.add_lag(move_done - turn_start - think)
                    if self.clock.left[self.my_color] < 0:
                        log("Our clock estimate is below zero - we may have flagged.")
                if label != "mine":
                    self._accept_opponent(label, fg, gray, time.time())
                return
            if self.ctl.interrupted():
                return
            log("Move didn't show up on the board - trying again...")
        log(f"Could not play the move. Paused - make the move by hand, then press {self.ctl.names[0]}.")
        self.board = after
        self.ctl.paused = True
        self.wait_until_resumed()
        if self.clock:
            self.clock.end_turn(self.my_color)

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

    def _accept_opponent(self, move, fg, gray, at=None):
        """at = when the move first showed up (stops the opponent's clock)."""
        before = self.board.copy()
        if self.clock:
            self.clock.end_turn(before.turn, at)
        self.board.push(move)
        log(f"Opponent played {before.san(move)}{self._clock_text()}")
        self._adapt(fg, gray, before, move)

    def their_turn(self):
        confirm, seen = None, 0
        mismatch_since = None
        while not self.ctl.interrupted():
            self.wait_until_resumed()
            if self.ctl.interrupted():
                return
            read_at = time.time()
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
                    log("The screen doesn't match the game I'm tracking (board moved? camera moved? "
                        "new game?). " + self.ctl.help() + ", or re-run calibration.")
                    mismatch_since = time.time() + 20
            else:
                mismatch_since = None

            if label is not None and none_cost - cost > MARGIN:
                if confirm != label:
                    first_seen = read_at
                seen = seen + 1 if confirm == label else 1
                confirm = label
                if seen >= self.args.confirm:        # same answer N reads in a row -> accept
                    self._accept_opponent(label, fg, gray, first_seen)
                    return
            else:
                confirm, seen = None, 0
            time.sleep(POLL)

    def play_game(self):
        self.ctl.new_game = False
        self.clock = timecontrol.Clock(*self.args.tc) if self.args.tc else None
        if self.clock:
            base, inc = self.args.tc
            log(f"Time control {base / 60:g}+{inc:g} - Stockfish manages its own clock.")
        log("Playing! " + self.ctl.help())
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
    p.add_argument("--mode", choices=list(timecontrol.MODES),
                   help="time control preset: bullet 1+0, blitz 3+2, rapid 10+0, classical 30+0")
    p.add_argument("--tc", help="custom time control, minutes+increment (e.g. 3+2, 0.5+0)")
    p.add_argument("--fen", help="start from this position (needs a profile with learned pieces)")
    p.add_argument("--input", choices=["auto", "drag", "click"], default="auto",
                   help="how to move pieces: auto (drag, fall back to click-click), drag or click")
    p.add_argument("--mouse", choices=["auto", "sendinput", "pynput"], default="auto",
                   help="mouse backend: sendinput = real input events (Windows default)")
    p.add_argument("--game", action="store_true",
                   help="preset for game clients: slower, frame-friendly input, tolerant of "
                        "animated scenes, hotkeys moved off F8-F10")
    p.add_argument("--settle", type=float, help="seconds to hover before pressing (default 0.05)")
    p.add_argument("--hold", type=float, help="seconds the button stays down on a click (default 0.04)")
    p.add_argument("--drag-time", type=float, help="seconds a drag takes (default 0.2)")
    p.add_argument("--click-gap", type=float, help="seconds between the two clicks of a move (default 0.08)")
    p.add_argument("--move-timeout", type=float, help="seconds to wait for our move to appear (default 3)")
    p.add_argument("--stable-tol", type=float, help="how still the board must be to read it (default 3)")
    p.add_argument("--confirm", type=int, help="identical reads needed to accept a move (default 2)")
    p.add_argument("--keys", help="hotkeys pause,newgame,quit (default f8,f10,f9)")
    p.add_argument("--engine", help="path to a UCI engine (default: engine/stockfish.exe)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--hash", type=int, default=128, help="engine hash size in MB")
    args = p.parse_args()
    # explicit flags win over the preset, the preset wins over the defaults
    preset = dict(settle=0.12, hold=0.08, drag_time=0.45, click_gap=0.3, move_timeout=6.0,
                  stable_tol=6.0, confirm=3, keys="f6,f7,end")
    normal = dict(settle=0.05, hold=0.04, drag_time=0.2, click_gap=0.08, move_timeout=3.0,
                  stable_tol=3.0, confirm=2, keys="f8,f10,f9")
    args.tc = timecontrol.parse_tc(args.tc) if args.tc else timecontrol.MODES.get(args.mode)
    if args.tc and args.tc[0] <= 90 and not args.game:
        # very fast games: spend less time on each mouse action so lag doesn't eat the clock
        normal.update(settle=0.02, hold=0.03, drag_time=0.12, click_gap=0.05)
    for k, v in (preset if args.game else normal).items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    Chessinator(args).run()


if __name__ == "__main__":
    main()
