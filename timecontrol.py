"""Time controls: presets, parsing, and our own record of both clocks.

The on-screen clock isn't read; instead both clocks are tracked from when moves
are made/seen, which is accurate to within the detection delay (a few tenths
of a second). Stockfish then manages its own thinking time from them.
"""
import statistics
import time

import chess

MODES = {                 # base seconds, increment seconds
    "bullet": (60, 0),
    "blitz": (180, 2),
    "rapid": (600, 0),
    "classical": (1800, 0),
}


def parse_tc(text):
    """'3+2' -> (180, 2);  '0.5+0' -> (30, 0);  '10' -> (600, 0). Minutes + seconds."""
    base, _, inc = text.replace(" ", "").partition("+")
    try:
        return float(base) * 60, float(inc or 0)
    except ValueError:
        raise SystemExit(f"Bad time control '{text}' - use minutes+increment, e.g. 3+2")


def fmt(sec):
    sec = max(0.0, sec)
    if sec < 20:
        return f"{sec:.1f}s"
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


class Clock:
    def __init__(self, base, inc):
        self.base, self.inc = base, inc
        self.left = {chess.WHITE: float(base), chess.BLACK: float(base)}
        self.turn_started = time.time()
        self.lag = []                 # our own non-thinking delay per move (seconds)

    def start(self, at=None):
        self.turn_started = at if at is not None else time.time()

    def end_turn(self, color, at=None):
        """The side `color` just moved at time `at`; charge them and add increment."""
        at = at if at is not None else time.time()
        self.left[color] -= max(0.0, at - self.turn_started)
        self.left[color] += self.inc
        self.turn_started = at

    def remaining(self, color, to_move):
        """Time left for `color` right now (the side to move is still ticking)."""
        left = self.left[color]
        if color == to_move:
            left -= time.time() - self.turn_started
        return left

    def add_lag(self, seconds):
        self.lag.append(max(0.0, seconds))
        del self.lag[:-12]

    def expected_lag(self, default_s):
        """Typical time per move spent outside the engine (spotting, mouse, confirming)."""
        return statistics.median(self.lag) if self.lag else default_s

    def budget(self, board, lag):
        """Seconds the engine may think on this move.

        Keeps a reserve for our own lag on every move still to come, spreads the
        rest over the expected remaining moves, and spends most of the increment.
        """
        mine = self.remaining(board.turn, board.turn)
        moves_left = max(15, 45 - board.fullmove_number // 2)
        usable = mine - lag * moves_left - 0.5          # this move's lag is in the reserve too
        think = usable / moves_left + 0.8 * self.inc
        return float(min(max(0.02, think), max(0.02, mine * 0.2)))
