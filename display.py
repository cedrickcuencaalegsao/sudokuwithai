"""In-place terminal animation: one pair of grids that is redrawn, never reprinted."""

from __future__ import annotations

import sys

from sudoku import Grid, format_side_by_side


class LiveView:
    """Draws `puzzle | current grid` plus a status line, then overwrites it on every update.

    Works by moving the cursor back up over the lines it just printed (ANSI escape codes),
    so the terminal only ever shows a single pair of tables. Use as a context manager so the
    cursor is restored even on Ctrl+C.
    """

    def __init__(self, puzzle: Grid) -> None:
        self.puzzle = puzzle
        self._lines = 0

    def __enter__(self) -> "LiveView":
        sys.stdout.write("\033[?25l")  # hide cursor
        sys.stdout.flush()
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.flush()

    def draw(self, grid: Grid, status: str, highlight: int | None = None, title: str = "Solving") -> None:
        frame = format_side_by_side(
            self.puzzle, grid, "Puzzle", title, original=self.puzzle, highlight=highlight
        )
        lines = (frame + "\n" + status).split("\n")
        out = []
        if self._lines:
            out.append(f"\033[{self._lines}A")  # back to the first line of the previous frame
        out.extend(f"\r\033[K{line}\n" for line in lines)
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._lines = len(lines)