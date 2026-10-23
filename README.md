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
python chessinator.py --game              # game clients: frame-friendly input, hotkeys F6/F7/End
python chessinator.py --input click       # force click-click (default "auto" figures it out)
python chessinator.py --mode blitz        # timed game: bullet, blitz, rapid or classical
python chessinator.py --tc 5+3            # any time control, minutes+increment
```

1. **Calibrate.** The console minimises itself so it can't cover the board, then a window
   shows the board it found with a grid on top, plus 4 zoomed corner views. Line the grid
   up with the squares exactly:
   - drag inside the box to move it, drag a corner to resize it
   - arrow keys = move 1px, `W/A/S/D` = move 10px, `[` / `]` = shrink/grow 1px
   - `M` = manual (click the board's top-left, then bottom-right corner)
   - `P` = perspective mode for tilted/3D boards: every corner moves on its own
     (`M` then asks for all 4 corners; `1`-`4` select a corner for the arrow keys;
     `O` rotates the grid 90 degrees)
   - `R` = auto-detect again, `Z` = show the whole screen
   - **ENTER** to accept
2. **Learn the pieces.** Show the starting position (as Black, it's fine if White has
   already made the first move) and press ENTER in the console. It figures out which side
   you are (the side at the bottom) and prints the board it sees. Press ENTER if it's right,
   or `f` to flip.
3. **Play.** Leave the board visible and don't cover it. The bot moves for you and watches
   for the opponent's replies.

| Hotkey | `--game` | Action |
|---|---|---|
| F8 | F6 | pause / resume (e.g. to take over by hand) |
| F10 | F7 | start a new game (re-learns pieces from the start position) |
| F9 | End | quit |

Pick your own with `--keys pause,newgame,quit` (e.g. `--keys f6,f7,end`).

## Making moves in any app

Moves are made with real mouse input (Windows `SendInput`): the cursor travels to the
piece, hovers, presses, moves in small steps across several frames, and releases. Apps
that read raw mouse input (most games) need this; they ignore a cursor that just jumps.

- `--input auto` (default) drags first. If the move doesn't show up, it retries with
  click-click and sticks with whichever one worked.
- `--game` slows input down for apps that process clicks once per frame or wait on a
  server: longer hover/hold, slower drags, a pause between the two clicks. It also
  tolerates animated scenery around the board and needs 3 matching reads before accepting
  a move.
- Fine-tune with `--settle`, `--hold`, `--drag-time`, `--click-gap`, `--move-timeout`
  (seconds), `--stable-tol` and `--confirm`.
- Keep the app's camera still once calibrated. If the view moves, recalibrate.

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

## Timed games (bullet / blitz / rapid)

Without a time control, every move gets the level's fixed think time above. With one,
the bot manages its clock so it doesn't flag:

| Mode | Time control |
|---|---|
| `--mode bullet` | 1+0 |
| `--mode blitz` | 3+2 |
| `--mode rapid` | 10+0 |
| `--mode classical` | 30+0 |
| `--tc M+S` | anything, e.g. `--tc 5+3`, `--tc 0.5+0` |

- The on-screen clock isn't read. Both clocks are tracked from when moves are made and
  seen, which is accurate to a few tenths of a second (on the safe side). The console
  shows both after every move.
- Each move's think time = (time left - a reserve for our own lag on every move still to
  come) / expected moves left + most of the increment. The lag (spotting the move, moving
  the mouse, confirming) is measured during the game, so slow apps get a bigger reserve.
- The level still applies: its depth cap and skill stay, so lower levels play fast anyway
  and level 8 tops out at depth 22.
- In very fast games (1 minute or less) mouse actions are sped up too, unless you set the
  timing flags yourself or use `--game`.
- Start the bot before or at the first move so its clocks line up with the game's.

## How it works

- **Board detection:** a multi-scale search for an 8x8 light/dark pattern, then pixel-accurate
  refinement by fitting the 7 inner grid lines to image edges. The board is stored as 4
  corners and read through a perspective warp, so tilted boards work the same way as flat
  ones once their corners are set.
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
