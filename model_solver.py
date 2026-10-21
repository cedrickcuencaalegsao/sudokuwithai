"""Solve sudoku cell by cell with a Tensor Loom ONNX model.

How the model and the rules share the work
------------------------------------------
* The model proposes: it sees the whole grid and scores every digit for every cell.
* The rules dispose: a digit is only eligible if it does not repeat in the cell's
  row, column or 3x3 box. Illegal digits are masked out and the rest renormalised.
* Each step fills the single cell the (masked) model is most confident about,
  then asks the model again with the updated grid.
* If a cell ends up with no legal digit, the solver backtracks and tries the
  next-best digit the model ranked for the most recent decision.

The model never gets to place an illegal digit, but a weak model can still
steer the search into dead ends. The summary reports how often that happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import onnxruntime as ort

from sudoku import Grid, candidates, is_solved

DIGITS = np.arange(1, 10, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class SudokuModel:
    """Wraps an ONNX model and turns any supported layout into a (81, 9) probability table.

    Supported inputs  (auto-detected from the model's input size):
      * 81 values  -> `input_encoding` "raw" (0-9) or "scaled" (0-1, value / 9)
      * 729 values -> one-hot, 9 channels per cell (empty cell = all zeros)
      * 810 values -> one-hot, 10 channels per cell (channel 0 = empty, 1-9 = digits)

    Supported outputs (auto-detected from the model's output size):
      * 81 values  -> regression, one number per cell, `output_encoding` "raw" (1-9) or "scaled"
      * 729 values -> per-cell digit scores, 9 channels per cell
      * 810 values -> per-cell scores, 10 channels per cell (channel 0 is ignored)

    One-hot / score layouts are cell-major by default (index = cell * channels + channel).
    Pass layout="channel-major" if yours is (index = channel * 81 + cell).
    """

    def __init__(
        self,
        path: str,
        input_encoding: str = "raw",
        output_encoding: str = "raw",
        temperature: float = 3.0,
        layout: str = "cell-major",
    ) -> None:
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.output_name = out.name
        self.in_size = int(inp.shape[-1])
        self.out_size = int(out.shape[-1])
        self.input_encoding = input_encoding
        self.output_encoding = output_encoding
        self.temperature = temperature
        self.channel_major = layout == "channel-major"

        if self.in_size not in (81, 729, 810):
            raise ValueError(f"Unsupported model input size {self.in_size}; expected 81, 729 or 810.")
        if self.out_size not in (81, 729, 810):
            raise ValueError(f"Unsupported model output size {self.out_size}; expected 81, 729 or 810.")

    # -- encoding ----------------------------------------------------------

    def _encode(self, grid: Grid) -> np.ndarray:
        g = np.asarray(grid, dtype=np.float32)
        if self.in_size in (729, 810):
            if self.in_size == 810:   # 10 channels: 0 = empty, 1-9 = digit
                x = np.zeros((81, 10), dtype=np.float32)
                for i, v in enumerate(grid):
                    x[i, v] = 1.0
            else:                      # 9 channels: empty cell = all zeros
                x = np.zeros((81, 9), dtype=np.float32)
                for i, v in enumerate(grid):
                    if v:
                        x[i, v - 1] = 1.0
            flat = x.T.reshape(-1) if self.channel_major else x.reshape(-1)
            return flat.reshape(1, self.in_size)
        if self.input_encoding == "scaled":
            g = g / 9.0
        return g.reshape(1, 81)

    # -- decoding ----------------------------------------------------------

    def predict(self, grid: Grid) -> np.ndarray:
        """Returns an (81, 9) array; row i holds the model's probability for digits 1..9 in cell i."""
        raw = self.session.run([self.output_name], {self.input_name: self._encode(grid)})[0]
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)

        if self.out_size == 81:
            pred = raw * 9.0 if self.output_encoding == "scaled" else raw
            scores = -np.abs(pred[:, None] - DIGITS[None, :])
            return _softmax(scores * self.temperature)

        channels = self.out_size // 81
        table = raw.reshape(channels, 81).T if self.channel_major else raw.reshape(81, channels)
        looks_like_probs = (
            table.min() >= 0.0 and table.max() <= 1.0 and np.allclose(table.sum(axis=1), 1.0, atol=1e-2)
        )
        if channels == 10:
            table = table[:, 1:]       # drop the "empty" channel; digits 1-9 remain
        if looks_like_probs:
            return table / np.maximum(table.sum(axis=1, keepdims=True), 1e-12)
        return _softmax(table)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@dataclass
class Step:
    cell: int
    digit: int
    confidence: float          # model confidence after rule masking (1.0 when forced)
    source: str                # "rules" (only one legal digit) or "model"
    correct: bool | None       # vs. known solution, None if unknown
    retry: bool = False        # True when placed after a backtrack

    @property
    def rc(self) -> str:
        return f"r{self.cell // 9 + 1}c{self.cell % 9 + 1}"


@dataclass
class SolveResult:
    grid: Grid
    solved: bool
    steps: list[Step] = field(default_factory=list)
    backtracks: int = 0
    inferences: int = 0
    gave_up: bool = False


def _rank_cell(probs_row: np.ndarray, legal: list[int]) -> tuple[list[int], float]:
    """Order the legal digits by model probability; also return the masked top confidence."""
    p = np.array([probs_row[d - 1] for d in legal], dtype=np.float64)
    total = p.sum()
    p = p / total if total > 1e-12 else np.full(len(legal), 1.0 / len(legal))
    order = np.argsort(-p)
    return [legal[i] for i in order], float(p[order[0]])


def solve_with_model(
    puzzle: Grid,
    model: SudokuModel,
    solution: Grid | None = None,
    backtrack: bool = True,
    max_backtracks: int = 20000,
    on_step: Callable[[Grid, Step, "SolveResult"], None] | None = None,
) -> SolveResult:
    grid = puzzle.copy()
    result = SolveResult(grid=grid, solved=False)

    # Each frame: (grid before the move, step count before the move, cell, digits not yet tried)
    stack: list[tuple[Grid, int, int, list[int]]] = []

    while True:
        empties = [i for i, v in enumerate(grid) if v == 0]
        if not empties:
            break

        # 1. Rules: legal digits for every empty cell. An empty set means a dead end.
        legal_by_cell = {i: candidates(grid, i) for i in empties}
        dead_end = any(not legal for legal in legal_by_cell.values())

        if not dead_end:
            # 2. Model: one inference per step, then pick the most confident cell.
            probs = model.predict(grid)
            result.inferences += 1

            best = None  # (confidence, -n_legal, cell, ranked)
            for i, legal in legal_by_cell.items():
                ranked, conf = _rank_cell(probs[i], legal)
                key = (conf, -len(legal))
                if best is None or key > best[0]:
                    best = (key, i, ranked)
            (conf, _), cell, ranked = best

            digit, rest = ranked[0], ranked[1:]
            stack.append((grid.copy(), len(result.steps), cell, rest))
            _place(grid, result, cell, digit, conf, len(legal_by_cell[cell]), solution, on_step)
            continue

        # 3. Dead end: undo the latest decision that still has alternatives.
        if not backtrack or result.backtracks >= max_backtracks:
            result.gave_up = True
            break

        resumed = False
        while stack:
            prev_grid, n_steps, cell, rest = stack.pop()
            if not rest:
                continue
            grid[:] = prev_grid
            del result.steps[n_steps:]
            result.backtracks += 1
            digit, rest = rest[0], rest[1:]
            stack.append((prev_grid, n_steps, cell, rest))
            _place(grid, result, cell, digit, 0.0, 2, solution, on_step, retry=True)
            resumed = True
            break
        if not resumed:
            result.gave_up = True
            break

    result.solved = is_solved(grid)
    return result


def _place(
    grid: Grid,
    result: SolveResult,
    cell: int,
    digit: int,
    conf: float,
    n_legal: int,
    solution: Grid | None,
    on_step: Callable[[Grid, Step, SolveResult], None] | None,
    retry: bool = False,
) -> None:
    grid[cell] = digit
    step = Step(
        cell=cell,
        digit=digit,
        confidence=conf,
        source="rules" if n_legal == 1 else "model",
        correct=None if solution is None else solution[cell] == digit,
        retry=retry,
    )
    result.steps.append(step)
    if on_step:
        on_step(grid, step, result)


# ---------------------------------------------------------------------------
# Model-only quality check (no rules, no search)
# ---------------------------------------------------------------------------

def one_shot_accuracy(model: SudokuModel, puzzle: Grid, solution: Grid) -> tuple[float, float]:
    """Single forward pass on the untouched puzzle.

    Returns (raw accuracy, rule-masked accuracy) over the empty cells: how often the
    model's top digit is right on its own, and how often it is right once illegal
    digits are removed. Handy for judging the model independent of the search.
    """
    probs = model.predict(puzzle)
    empties = [i for i, v in enumerate(puzzle) if v == 0]
    raw_hits = masked_hits = 0
    for i in empties:
        if int(np.argmax(probs[i])) + 1 == solution[i]:
            raw_hits += 1
        legal = candidates(puzzle, i)
        if legal:
            ranked, _ = _rank_cell(probs[i], legal)
            masked_hits += ranked[0] == solution[i]
    n = max(1, len(empties))
    return raw_hits / n, masked_hits / n