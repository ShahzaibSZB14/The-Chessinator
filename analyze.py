"""Chessinator Live - real-time analysis of a game you play yourself.

Unlike chessinator.py (which plays the moves for you with the mouse), this never
touches the mouse. It watches the board on your screen and, after every move -
yours or the bot's - shows Stockfish's evaluation, the best move, the top lines,
and how good the move that was just played was. The best move is drawn as an
arrow straight onto your real board, with an evaluation bar beside it.

You play every move yourself on your own website; this just analyses as you go,
so you can play the best move and watch how the bots respond to it.

Usage:
    python analyze.py                      # analyse, overlay on, depth-limited
    python analyze.py --depth 18           # how deep Stockfish looks (default 16)
    python analyze.py --time 1.0           # seconds per position (with --depth, min wins)
    python analyze.py --lines 3            # how many candidate moves to show
    python analyze.py --color white        # force your side (default: auto-detect)
    python analyze.py --no-overlay         # console only, no on-screen drawing
    python analyze.py --profile lichess    # keep separate board calibrations per site

Hotkeys (defaults):  F8 pause/resume   F10 re-sync (new game)   F9 quit
"""
import argparse
import ctypes
import os
import pickle
import sys
import time

import chess
import chess.engine
import numpy as np

# Real pixels on scaled (125%/150%) displays - must run before capture/Tk start.
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
import overlay as overlay_mod  # noqa: E402
from chessinator import Controls, find_engine, ascii_board, log  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, "profiles")

MARGIN = 0.12            # a move must explain the screen this much better than "nothing moved"
POLL = 0.08
MATE_CP = 100000         # centipawn stand-in for a forced mate when comparing evals


# --------------------------------------------------------------------------
# Evaluation helpers (everything is kept in centipawns from White's point of view)
# --------------------------------------------------------------------------

def white_cp(score):
    """info['score'] (a PovScore) -> centipawns from White's side (mate = +-MATE_CP)."""
    return score.white().score(mate_score=MATE_CP)


def white_mate(score):
    """Moves-to-mate from White's side, or None if it isn't a forced mate."""
    return score.white().mate()


def fmt_eval(cp, mate):
    if mate is not None:
        return f"#{mate}" if mate > 0 else f"#-{abs(mate)}"
    return f"{cp / 100:+.2f}"


def classify(loss_cp, played_best):
    """Rate a move by how many centipawns worse it was than the best move."""
    if played_best or loss_cp <= 10:
        return "best", "best move"
    if loss_cp <= 40:
        return "good", "good"
    if loss_cp <= 90:
        return "inaccuracy", "inaccuracy ?!"
    if loss_cp <= 250:
        return "mistake", "mistake ?"
    return "blunder", "blunder ??"


# --------------------------------------------------------------------------
# Engine (analysis only)
# --------------------------------------------------------------------------

class Analyst:
    def __init__(self, path, depth, movetime, multipv, threads, hash_mb):
        self.depth, self.movetime, self.multipv = depth, movetime, multipv
        self.eng = chess.engine.SimpleEngine.popen_uci(path)
        opts = {"Threads": threads, "Hash": hash_mb}
        self.eng.configure({k: v for k, v in opts.items() if k in self.eng.options})

    def analyse(self, board):
        """Return a list of info dicts (one per line), best line first."""
        limit = chess.engine.Limit(depth=self.depth, time=self.movetime)
        info = self.eng.analyse(board, limit, multipv=self.multipv)
        return info if isinstance(info, list) else [info]

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Live analyzer
# --------------------------------------------------------------------------

class LiveAnalyzer:
    def __init__(self, args):
        self.args = args
        self.screen = vision.Screen()
        self.ctl = Controls(args.keys)
        self.profile_path = os.path.join(PROFILES, args.profile + ".pkl")
        self.profile = self._load_profile()
        self.quad = None
        self.rec = None
        self.board = None
        self.my_color = None
        self.cur_cp = 0            # eval (White cp) of the position on the board now
        self.cur_mate = None
        self.cur_best = None       # best move in the position on the board now
        self.engine = Analyst(find_engine(args.engine), args.depth, args.time,
                              max(1, args.lines), args.threads, args.hash)
        self.overlay = overlay_mod.Overlay() if args.overlay else None
        if args.overlay and self.overlay and not self.overlay.ok:
            log("Overlay unavailable - showing analysis in the console only.")
            self.overlay = None

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
    def square_center(self, square):
        row, col = divmod(self.rec.screen_index(square), 8)
        return vision.board_to_screen(self.quad, (col + 0.5) / 8, (row + 0.5) / 8)

    # ---- vision helpers ------------------------------------------------
    def stable_frame(self, timeout=None):
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
        scored = []
        for label, b in cands:
            c, cls = self.rec.cost(D, b)
            scored.append((c, label, b, cls))
        scored.sort(key=lambda s: s[0])
        c, label, b, cls = scored[0]
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
        with console.hidden():
            quad = calibrate.calibrate(self.screen, quad)
        if quad is None:
            sys.exit("Calibration cancelled.")
        self.quad = quad
        log("Board corners: " + "  ".join(f"({x:.0f},{y:.0f})" for x, y in quad))

    def setup_game(self):
        if self.args.fen:
            st = self.profile.get("recognizer")
            if not st:
                sys.exit("--fen needs a profile that was calibrated on a starting position first.")
            self.rec = vision.Recognizer()
            self.rec.load_state(st)
            if self.args.color:
                self.rec.white_bottom = self.args.color == "white"
            self.board = chess.Board(self.args.fen)
            self.args.fen = None
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

        self.my_color = chess.WHITE if self.rec.white_bottom else chess.BLACK
        print()
        print(ascii_board(self.board, self.rec.white_bottom))
        print(f"\n    You are {'WHITE' if self.my_color else 'BLACK'} (bottom side).")
        ans = input("    Correct?  [ENTER = yes / f = flip colours / q = quit] ").strip().lower()
        if ans == "q":
            sys.exit(0)
        if ans == "f":
            self.args.color = "black" if self.my_color == chess.WHITE else "white"
            return self.setup_game()
        self._save_profile()

    # ---- analysis + display --------------------------------------------
    def _adapt(self, fg, gray, before, move):
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

    def _pv_san(self, board, pv, limit=5):
        b = board.copy()
        out = []
        for mv in pv[:limit]:
            if mv not in b.legal_moves:
                break
            out.append(b.san(mv))
            b.push(mv)
        return out

    def analyze_position(self):
        """Analyse the position currently on the board; update console + overlay."""
        if self.board.is_game_over():
            self.cur_best = None
            res = self.board.result()
            term = self.board.outcome().termination.name.replace("_", " ").lower()
            log(f"Game over: {res} ({term})")
            if self.overlay:
                self.overlay.set_state(quad=self.quad, arrow=None,
                                       white_bottom=self.rec.white_bottom,
                                       status=f"Game over  {res}", lines=[term])
            return

        infos = self.engine.analyse(self.board)
        best_info = infos[0]
        self.cur_best = best_info["pv"][0]
        self.cur_cp = white_cp(best_info["score"])
        self.cur_mate = white_mate(best_info["score"])

        turn = "White" if self.board.turn == chess.WHITE else "Black"
        to_move = "you" if self.board.turn == self.my_color else "opponent"
        header = f"{turn} to move ({to_move}) | eval {fmt_eval(self.cur_cp, self.cur_mate)}"
        lines = []
        for i, info in enumerate(infos):
            san = self._pv_san(self.board, info.get("pv", []))
            if not san:
                continue
            tag = fmt_eval(white_cp(info["score"]), white_mate(info["score"]))
            marker = "->" if i == 0 else "  "
            lines.append(f"{marker} {tag:>7}  {' '.join(san)}")
        print()
        log(header)
        for ln in lines:
            print("        " + ln)

        if self.overlay:
            a = self.square_center(self.cur_best.from_square)
            b = self.square_center(self.cur_best.to_square)
            self.overlay.set_state(
                quad=self.quad, arrow=(a[0], a[1], b[0], b[1]),
                eval_white=self.cur_cp, mate=self.cur_mate,
                white_bottom=self.rec.white_bottom,
                status=header, lines=lines)

    def report_move(self, mover, san, before_cp, before_mate, before_best, move):
        """Print how good the move that was just played was."""
        # Loss = best the mover could have had, minus what they got (mover's point of view).
        sign = 1 if mover == chess.WHITE else -1
        before = sign * before_cp
        after = sign * self.cur_cp
        loss = max(0, before - after)
        played_best = (before_best is not None and move == before_best)
        _, label = classify(loss, played_best)
        who = "You" if mover == self.my_color else "Opponent"
        # Only show a centipawn cost for normal (non-mate) positions.
        show_cost = not played_best and before_mate is None and self.cur_mate is None
        extra = f"  (-{loss / 100:.2f})" if show_cost else ""
        log(f"{who} played {san}  ->  {label}{extra}")

    # ---- main loop -----------------------------------------------------
    def wait_until_resumed(self):
        while self.ctl.paused and not self.ctl.interrupted():
            time.sleep(0.1)

    def watch_move(self):
        """Wait until the side to move makes a move on screen; return it (or None)."""
        confirm, seen, first_seen = None, 0, None
        mismatch_since = None
        while not self.ctl.interrupted():
            self.wait_until_resumed()
            if self.ctl.interrupted():
                return None
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
                    log("The screen no longer matches the game I'm tracking (new game? "
                        "take-back? board moved?). " + self.ctl.help())
                    mismatch_since = time.time() + 20
            else:
                mismatch_since = None

            if label is not None and none_cost - cost > MARGIN:
                if confirm != label:
                    first_seen = time.time()
                seen = seen + 1 if confirm == label else 1
                confirm = label
                if seen >= self.args.confirm:
                    return label, fg, gray
            else:
                confirm, seen = None, 0
            time.sleep(POLL)
        return None

    def play_game(self):
        self.ctl.new_game = False
        log("Watching your game. " + self.ctl.help())
        self.analyze_position()                 # show the best move for the opening position
        while not self.ctl.interrupted():
            self.wait_until_resumed()
            if self.board.is_game_over():
                return
            res = self.watch_move()
            if res is None:
                return
            move, fg, gray = res
            mover = self.board.turn
            before_cp, before_mate, before_best = self.cur_cp, self.cur_mate, self.cur_best
            san = self.board.san(move)
            before_board = self.board.copy()
            self.board.push(move)
            self._adapt(fg, gray, before_board, move)
            self.analyze_position()             # eval of the new position (sets self.cur_*)
            if not self.board.is_game_over():
                self.report_move(mover, san, before_cp, before_mate, before_best, move)

    def run(self):
        try:
            self.setup_board_rect()
            while not self.ctl.quit:
                self.setup_game()
                self.play_game()
                if self.ctl.quit:
                    break
                if not self.ctl.new_game:
                    ans = input("\n>>> Analyse another game? [ENTER = yes / q = quit] ").strip().lower()
                    if ans == "q":
                        break
        finally:
            self.engine.close()
            if self.overlay:
                self.overlay.close()


def main():
    p = argparse.ArgumentParser(
        description="Chessinator Live - real-time analysis of a game you play yourself "
                    "(shows the best move, never touches the mouse).")
    p.add_argument("--depth", type=int, default=16, help="search depth per position (default 16)")
    p.add_argument("--time", type=float, default=0.6,
                   help="max seconds per position; whichever of depth/time is reached first "
                        "ends the search (default 0.6)")
    p.add_argument("--lines", type=int, default=3, help="candidate moves to show (MultiPV, default 3)")
    p.add_argument("--profile", default="default",
                   help="name to save/load board calibration under (e.g. lichess, chesscom)")
    p.add_argument("--skip-calibration", action="store_true",
                   help="reuse the saved board position without opening the calibration window")
    p.add_argument("--color", choices=["white", "black"],
                   help="force your side (default: auto-detect, bottom side)")
    p.add_argument("--fen", help="start from this position (needs a profile with learned pieces)")
    p.add_argument("--no-overlay", dest="overlay", action="store_false",
                   help="don't draw on screen; print analysis in the console only")
    p.add_argument("--stable-tol", type=float, default=3.0,
                   help="how still the board must be before it's read (default 3)")
    p.add_argument("--confirm", type=int, default=2,
                   help="identical reads needed to accept a move (default 2)")
    p.add_argument("--keys", default="f8,f10,f9",
                   help="hotkeys pause,newgame,quit (default f8,f10,f9)")
    p.add_argument("--engine", help="path to a UCI engine (default: engine/stockfish.exe)")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--hash", type=int, default=128, help="engine hash size in MB")
    args = p.parse_args()
    LiveAnalyzer(args).run()


if __name__ == "__main__":
    main()
