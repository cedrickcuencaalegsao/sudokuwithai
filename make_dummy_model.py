"""Writes a random-weight .onnx model so you can smoke-test the pipeline.

The model is NOT trained, so expect poor model choices (the solver will lean on the
rules and backtracking). Replace it with the .onnx exported from Tensor Loom.

  python make_dummy_model.py --out dummy81.onnx               # 81 -> 81 regression-style
  python make_dummy_model.py --out dummy729.onnx --classes    # 81 -> 729 softmax-style
"""

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(in_size: int, out: int, hidden: int = 256) -> onnx.ModelProto:
    classes = out != 81
    rng = np.random.default_rng(0)
    w1 = (rng.standard_normal((in_size, hidden)) * 0.1).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = (rng.standard_normal((hidden, out)) * 0.1).astype(np.float32)
    b2 = (np.full(out, 0.0 if classes else 5.0)).astype(np.float32)

    nodes = [
        helper.make_node("Gemm", ["input", "W0", "B0"], ["fc0"]),
        helper.make_node("Relu", ["fc0"], ["act0"]),
        helper.make_node("Gemm", ["act0", "W1", "B1"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", in_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", out])],
        [numpy_helper.from_array(a, n) for a, n in [(w1, "W0"), (b1, "B0"), (w2, "W1"), (b2, "B1")]],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--in-size", type=int, default=81, choices=[81, 729, 810])
    ap.add_argument("--out-size", type=int, default=81, choices=[81, 729, 810])
    ap.add_argument("--classes", action="store_true", help="shortcut for --out-size 729")
    a = ap.parse_args()
    onnx.save(build(a.in_size, 729 if a.classes else a.out_size), a.out)
    print("wrote", a.out)