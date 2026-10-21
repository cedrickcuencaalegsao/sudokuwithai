# sudoku-onnx-solver

Generates a sudoku puzzle, then solves it **cell by cell** with an ONNX model exported
from Tensor Loom, enforcing the sudoku rules on every move.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Make shortcuts

```bash
make setup                        # create .venv, install dependencies
make run MODEL=../model.onnx      # random puzzle
make run CLUES=28 SEED=7          # harder, reproducible
make animate                      # redraw after every digit
make puzzle P="53..7....6..195...."
make run ARGS="--layout channel-major"   # pass any extra flag
make dummy                        # smoke test without a trained model
make clean
```

## Run

```bash
python main.py --model path/to/sudoku.onnx                      # random 32-clue puzzle
python main.py --model sudoku.onnx --clues 28 --seed 7 --animate
python main.py --model sudoku.onnx --puzzle "53..7....6..195...."   # your own 81-char puzzle
```

Smoke test without a trained model (random weights, so the model's own choices will be poor):

```bash
python make_dummy_model.py --out dummy.onnx
python main.py --model dummy.onnx
```

## How a step works

1. **Rules** list the legal digits for every empty cell (row / column / box).
2. **Model** scores digits for the whole grid; illegal digits are masked out.
3. The cell the masked model is most confident about gets its top digit.
4. Repeat with the updated grid. If a cell has no legal digit, backtrack and try the
   model's next-best digit for the latest decision (`--no-backtrack` disables this).

Steps are labelled `rules` (only one legal digit, so no model judgement needed) or `model`.
The summary shows how many model choices matched the true solution, plus one-shot accuracy
(a single forward pass, with and without the rules), which is the cleanest read on model quality.

## Matching your model's encoding

| Model | Flag |
|---|---|
| 81 inputs, digits 0-9 | `--input-encoding raw` (default) |
| 81 inputs, digit / 9 | `--input-encoding scaled` |
| 729 inputs | one-hot, 9 channels per cell (empty = all zeros), auto-detected |
| 810 inputs | one-hot, 10 channels per cell (channel 0 = empty), auto-detected |
| 81 outputs, digits 1-9 | `--output-encoding raw` (default) |
| 81 outputs, digit / 9 | `--output-encoding scaled` |
| 729 / 810 outputs | per-cell digit scores, 9 or 10 channels (channel 0 ignored), auto-detected |
| one-hot / score layout | cell-major (`cell * channels + channel`) by default, `--layout channel-major` for `channel * 81 + cell` |

Check the first row of the CSV the model was trained on to pick the right flags. If
one-shot accuracy is near 11% (random guessing among 9 digits), the encoding is probably wrong.# sudokuwithai
# sudokuwithai
