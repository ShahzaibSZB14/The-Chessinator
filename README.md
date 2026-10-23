# The Chessinator

Reads any chessboard on your screen, works out where every square is, recognises the
pieces, and plays the moves for you with Stockfish (level 8 = full strength by default).

It works with any board colours and piece set: it learns what *your* pieces look like
from the starting position, so nothing is hard-coded for a particular site or theme.

## Setup (once)

```
pip install -r requirements.txt
python setup_stockfish.py        # downloads Stockfish into engine\stockfish.exe
```

## Running

Double-click `Chessinator.bat`, or run:

```
python chessinator.py                     # level 8, auto-detect everything
python chessinator.py --level 5           # weaker (1-8)
python chessinator.py --profile lichess   # keep separate calibrations per site
python chessinator.py --input click       # click-click instead of drag-and-drop
```

1. **Calibrate.** A window shows the board it found with a grid on top, plus 4 zoomed
   corner views. Line the grid up with the squares exactly:
   - drag inside the box to move it, drag a corner to resize it
   - arrow keys = move 1px, `W/A/S/D` = move 10px, `[` / `]` = shrink/grow 1px
   - `M` = manual (click the board's top-left, then bottom-right corner)
   - `R` = auto-detect again, `Z` = show the whole screen
   - **ENTER** to accept
2. **Learn the pieces.** Show the starting position (as Black, it's fine if White has
   already made the first move) and press ENTER in the console. It figures out which side
   you are (the side at the bottom) and prints the board it sees. Press ENTER if it's right,
   or `f` to flip.
3. **Play.** Leave the board visible and don't cover it. The bot moves for you and watches
   for the opponent's replies.

| Hotkey | Action |
|---|---|
| F8 | pause / resume (e.g. to take over by hand) |
| F10 | start a new game (re-learns pieces from the start position) |
| F9 | quit |

The board position is saved in `profiles/<name>.pkl`. Next time, it opens already
lined up, so you only press ENTER. Use `--skip-calibration` to skip the window completely.

## Levels

Same scale as Lichess's "play with the computer" levels:

| Level | Skill Level | Depth | Think time |
|---|---|---|---|
| 1-3 | 0 | 5 | 50-150 ms |
| 4 | 3 | 5 | 200 ms |
| 5 | 7 | 5 | 300 ms |
| 6 | 11 | 8 | 400 ms |
| 7 | 16 | 13 | 500 ms |
| **8** | **20** | **22** | **1 s** |

## How it works

- **Board detection:** a multi-scale search for an 8x8 light/dark pattern, then pixel-accurate
  refinement by fitting the 7 inner grid lines to image edges.
- **Pieces:** each square becomes a signature measured against that square's own
  background colour, so square colour and move highlights matter very little. Coordinate
  letters in the corners are masked out. Signatures are learned from the starting position
  and keep adapting during the game.
- **Moves:** rather than trusting a raw per-square guess, it scores every *legal* move by how
  well it explains what's on screen. A move counts only once the screen has stopped changing
  (no mid-animation frames) and the same answer shows up twice in a row. This handles castling,
  en passant and promotions. If your own move doesn't register, it tries again.
- **Multi-monitor and display scaling (125%/150%)** are supported (it runs DPI-aware).

## Tips / troubleshooting

- Turn off piece-move animations or "premove" arrows if your site has unusual overlays;
  normal last-move highlights are fine.
- If you move/resize the browser window or zoom the page, re-run calibration (the bot warns
  when the screen stops matching the game it's tracking).
- Promotion assumes the usual picker layout (Queen on the target square, then Knight, Rook,
  Bishop toward the centre), as on Lichess and Chess.com.
- Resuming a game in progress: `--fen "<FEN>"` (needs a profile where pieces were learned).

## Fair play

Using engine assistance in rated games against people breaks the rules on Chess.com,
Lichess and every other site, and gets accounts banned. Use it against computer
opponents, in your own analysis/test boards, bot-vs-bot experiments, or casual games
where everyone agreed to it.
