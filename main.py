"""Generate a sudoku puzzle and solve it cell by cell with a Tensor Loom ONNX model.

Examples
--------
  python main.py --model sudoku.onnx
  python main.py --model sudoku.onnx --clues 36 --seed 7 --animate
  python main.py --model sudoku.onnx --puzzle "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
  python main.py --model sudoku.onnx --input-encoding scaled --output-encoding scaled
"""

from __future__ import annotations

import argparse
import sys
import time

from display import LiveView
from model_solver import SudokuModel, SolveResult, Step, one_shot_accuracy, solve_with_model
from sudoku import (
    count_solutions,
    format_grid,
    format_side_by_side,
    grid_to_string,
    make_puzzle,
    parse_grid,
    solve_backtracking,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to the exported .onnx model")
    p.add_argument("--clues", type=int, default=32, help="number of given cells in a generated puzzle (default 32)")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible puzzles")
    p.add_argument("--puzzle", help="81-char puzzle string (0 or . = empty) instead of generating one")
    p.add_argument("--input-encoding", choices=["raw", "scaled"], default="raw",
                   help="81-input models: feed digits 0-9 (raw) or digit/9 (scaled)")
    p.add_argument("--output-encoding", choices=["raw", "scaled"], default="raw",
                   help="81-output models: model outputs digits 1-9 (raw) or digit/9 (scaled)")
    p.add_argument("--layout", choices=["cell-major", "channel-major"], default="cell-major",
                   help="one-hot / score models: index = cell*channels + channel (default) or channel*81 + cell")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="regression models only: sharpness when turning a number into digit probabilities")
    p.add_argument("--no-backtrack", action="store_true", help="stop at the first dead end instead of backtracking")
    p.add_argument("--max-backtracks", type=int, default=20000)
    p.add_argument("--animate", action="store_true", help="watch it solve: one pair of grids redrawn in place (needs a real terminal)")
    p.add_argument("--delay", type=float, default=0.1, help="seconds between frames with --animate")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Puzzle -------------------------------------------------------------
    if args.puzzle:
        puzzle = parse_grid(args.puzzle)
        n = count_solutions(puzzle, 2)
        if n == 0:
            print("That puzzle has no solution.")
            return 1
        solution = solve_backtracking(puzzle)
        if n > 1:
            print("Note: that puzzle has more than one solution; correctness is judged against one of them.")
    else:
        puzzle, solution = make_puzzle(args.clues, args.seed)

    animate = args.animate and sys.stdout.isatty()
    if args.animate and not animate:
        print("(--animate needs an interactive terminal; falling back to the step log)\n")

    if not animate:
        print("Puzzle")
        print(format_grid(puzzle))
    print(f"({sum(1 for v in puzzle if v)} clues)  {grid_to_string(puzzle)}\n")

    # 2. Model --------------------------------------------------------------
    model = SudokuModel(args.model, args.input_encoding, args.output_encoding, args.temperature, args.layout)
    print(f"Model: {model.in_size} inputs -> {model.out_size} outputs "
          f"({'per-cell regression' if model.out_size == 81 else 'per-cell digit scores'}, "
          f"{'one-hot input' if model.in_size > 81 else 'numeric input'})")

    raw_acc, masked_acc = one_shot_accuracy(model, puzzle, solution)
    print(f"One-shot accuracy on empty cells: {raw_acc:.0%} raw, {masked_acc:.0%} after applying sudoku rules\n")

    # 3. Solve cell by cell -------------------------------------------------
    solve_kwargs = dict(backtrack=not args.no_backtrack, max_backtracks=args.max_backtracks)

    if animate:
        with LiveView(puzzle) as view:
            view.draw(puzzle, "starting...")

            def on_step(grid, step: Step, res: SolveResult) -> None:
                kind = "retry" if step.retry else step.source
                view.draw(
                    grid,
                    f"step {len(res.steps):>2}  {step.rc} = {step.digit}  {kind:<5} "
                    f"conf {step.confidence:.2f}  backtracks {res.backtracks}",
                    highlight=step.cell,
                )
                time.sleep(args.delay)

            result = solve_with_model(puzzle, model, solution, on_step=on_step, **solve_kwargs)
            view.draw(
                result.grid,
                f"{'solved' if result.solved else 'not solved'} in {len(result.steps)} placements, "
                f"{result.backtracks} backtracks",
                title="Solved" if result.solved else "Unsolved",
            )
        print()
    else:
        def on_step(grid, step: Step, res: SolveResult) -> None:
            mark = {True: "ok ", False: "BAD", None: "   "}[step.correct]
            print(f"  [{mark}] {step.rc} = {step.digit}   {step.source:<5}  conf {step.confidence:.2f}")

        print("Solving:")
        result = solve_with_model(puzzle, model, solution, on_step=on_step, **solve_kwargs)
        print("\nResult")
        print(format_side_by_side(puzzle, result.grid, "Puzzle", "Solved" if result.solved else "Unsolved", original=puzzle))

    # 4. Report -------------------------------------------------------------
    model_steps = [s for s in result.steps if s.source == "model"]
    rule_steps = [s for s in result.steps if s.source == "rules"]
    right = sum(1 for s in model_steps if s.correct)

    print(f"Solved: {'yes' if result.solved else 'NO'}"
          f"{'  (gave up)' if result.gave_up else ''}")
    print(f"Cells filled on final path: {len(result.steps)}  "
          f"({len(rule_steps)} forced by rules, {len(model_steps)} chosen by the model)")
    if model_steps:
        print(f"Model choices matching the true solution: {right}/{len(model_steps)} ({right / len(model_steps):.0%})")
    print(f"Backtracks: {result.backtracks}   Model inferences: {result.inferences}")
    return 0 if result.solved else 2


if __name__ == "__main__":
    sys.exit(main())