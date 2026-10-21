"""Sudoku rules, puzzle generation and helpers.

A grid is a flat list of 81 ints, row-major. 0 means empty, 1-9 are digits.
Everything rule-related lives here so the model-driven solver can ask
"is this digit legal in this cell?" without knowing how the check works.
"""

from __future__ import annotations

import random
import re

Grid = list[int]

ALL_DIGITS = 0b1111111110  # bits 1..9 set


def box_of(idx: int) -> int:
    r, c = divmod(idx, 9)
    return (r // 3) * 3 + c // 3


def _masks(grid: Grid) -> tuple[list[int], list[int], list[int]]:
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(grid):
        if v:
            bit = 1 << v
            rows[i // 9] |= bit
            cols[i % 9] |= bit
            boxes[box_of(i)] |= bit
    return rows, cols, boxes


def _bits_to_digits(mask: int) -> list[int]:
    return [d for d in range(1, 10) if mask & (1 << d)]


def candidates(grid: Grid, idx: int) -> list[int]:
    """Digits that can legally go in cell `idx` (empty list if it is filled)."""
    if grid[idx]:
        return []
    r, c = divmod(idx, 9)
    used = 0
    for k in range(9):
        used |= 1 << grid[r * 9 + k]
        used |= 1 << grid[k * 9 + c]
    br, bc = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            used |= 1 << grid[(br + dr) * 9 + bc + dc]
    return [d for d in range(1, 10) if not used & (1 << d)]


def is_consistent(grid: Grid) -> bool:
    """True if no row, column or box contains a repeated digit."""
    for group in _groups():
        seen = 0
        for i in group:
            v = grid[i]
            if v:
                if seen & (1 << v):
                    return False
                seen |= 1 << v
    return True


def is_solved(grid: Grid) -> bool:
    return all(grid) and is_consistent(grid)


def _groups() -> list[list[int]]:
    groups = [[r * 9 + c for c in range(9)] for r in range(9)]
    groups += [[r * 9 + c for r in range(9)] for c in range(9)]
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            groups.append([(br + dr) * 9 + bc + dc for dr in range(3) for dc in range(3)])
    return groups


# ---------------------------------------------------------------------------
# Fast backtracking solver (used for generation and uniqueness checks only;
# the ONNX model does the "real" solving in model_solver.py)
# ---------------------------------------------------------------------------

def _search(grid: Grid, rows, cols, boxes, rng: random.Random | None, limit: int, found: list) -> None:
    # Choose the empty cell with the fewest candidates (MRV).
    best_idx, best_mask, best_n = -1, 0, 10
    for i in range(81):
        if grid[i] == 0:
            r, c = divmod(i, 9)
            mask = ALL_DIGITS & ~(rows[r] | cols[c] | boxes[box_of(i)])
            n = bin(mask).count("1")
            if n < best_n:
                best_idx, best_mask, best_n = i, mask, n
                if n <= 1:
                    break
    if best_idx == -1:
        found.append(grid.copy())
        return
    if best_n == 0:
        return

    digits = _bits_to_digits(best_mask)
    if rng is not None:
        rng.shuffle(digits)
    r, c = divmod(best_idx, 9)
    b = box_of(best_idx)
    for d in digits:
        bit = 1 << d
        grid[best_idx] = d
        rows[r] |= bit
        cols[c] |= bit
        boxes[b] |= bit
        _search(grid, rows, cols, boxes, rng, limit, found)
        rows[r] &= ~bit
        cols[c] &= ~bit
        boxes[b] &= ~bit
        grid[best_idx] = 0
        if len(found) >= limit:
            return


def solve_backtracking(grid: Grid, rng: random.Random | None = None) -> Grid | None:
    g = grid.copy()
    rows, cols, boxes = _masks(g)
    found: list[Grid] = []
    _search(g, rows, cols, boxes, rng, 1, found)
    return found[0] if found else None


def count_solutions(grid: Grid, limit: int = 2) -> int:
    g = grid.copy()
    rows, cols, boxes = _masks(g)
    found: list[Grid] = []
    _search(g, rows, cols, boxes, None, limit, found)
    return len(found)


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------

def generate_solution(rng: random.Random) -> Grid:
    solved = solve_backtracking([0] * 81, rng)
    assert solved is not None
    return solved


def make_puzzle(clues: int = 32, seed: int | None = None) -> tuple[Grid, Grid]:
    """Returns (puzzle, solution). The puzzle has a unique solution.

    `clues` is a target; very low values (< ~24) may not be reachable while
    keeping uniqueness, in which case the puzzle simply ends up with more clues.
    """
    clues = max(17, min(80, clues))
    rng = random.Random(seed)
    solution = generate_solution(rng)
    puzzle = solution.copy()

    order = list(range(81))
    rng.shuffle(order)
    filled = 81
    for idx in order:
        if filled <= clues:
            break
        saved = puzzle[idx]
        puzzle[idx] = 0
        if count_solutions(puzzle, 2) != 1:
            puzzle[idx] = saved
        else:
            filled -= 1
    return puzzle, solution


# ---------------------------------------------------------------------------
# Parsing / printing
# ---------------------------------------------------------------------------

def parse_grid(text: str) -> Grid:
    """Accepts 81 chars of digits with 0 or . for empty (whitespace ignored)."""
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) != 81:
        raise ValueError(f"expected 81 cells, got {len(chars)}")
    grid = [0 if ch in "0._" else int(ch) for ch in chars]
    if not is_consistent(grid):
        raise ValueError("the given puzzle already breaks the sudoku rules")
    return grid


def grid_to_string(grid: Grid) -> str:
    return "".join(str(v) if v else "." for v in grid)


def format_grid(grid: Grid, highlight: int | None = None, original: Grid | None = None) -> str:
    """Pretty-print a grid. `highlight` marks one cell, `original` dims given clues."""
    lines = []
    sep = "+-------+-------+-------+"
    for r in range(9):
        if r % 3 == 0:
            lines.append(sep)
        parts = []
        for c in range(9):
            i = r * 9 + c
            v = grid[i]
            ch = str(v) if v else "."
            if i == highlight:
                ch = f"\033[1;32m{ch}\033[0m"
            elif original is not None and v and not original[i]:
                ch = f"\033[36m{ch}\033[0m"
            parts.append(ch)
            if c % 3 == 2:
                parts.append("|")
        lines.append("| " + " ".join(parts))
    lines.append(sep)
    return "\n".join(lines)


_ANSI = re.compile(r"\033\[[0-9;]*m")


def format_side_by_side(
    left: Grid,
    right: Grid,
    left_title: str = "Puzzle",
    right_title: str = "Solved",
    original: Grid | None = None,
    highlight: int | None = None,
    gap: str = "   ->   ",
) -> str:
    """Two grids next to each other. `right` shows digits the solver added in cyan."""
    left_lines = format_grid(left).split("\n")
    right_lines = format_grid(right, highlight=highlight, original=original).split("\n")
    width = len(_ANSI.sub("", left_lines[0]))
    head = left_title.center(width) + " " * len(gap) + right_title.center(width)
    rows = [head.rstrip()]
    for a, b in zip(left_lines, right_lines):
        pad = " " * (width - len(_ANSI.sub("", a)))
        rows.append(a + pad + gap + b)
    return "\n".join(rows)