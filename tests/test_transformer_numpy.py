"""Offline runtime checks for the NumPy Transformer notebook."""

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "03_Transformer_Encoder_From_Scratch.ipynb"


def load_definitions():
    namespace = {"np": np}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        definitions = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        if definitions:
            module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
            exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


class TransformerChecks(unittest.TestCase):
    def test_attention_probabilities_and_classifier_training(self):
        functions = load_definitions()
        rng = np.random.default_rng(42)
        x_train = rng.integers(0, 50, size=(12, 6))
        y_train = rng.integers(0, 2, size=12)
        x_val = rng.integers(0, 50, size=(4, 6))
        y_val = rng.integers(0, 2, size=4)

        model = functions["TransformerClassifier"](
            vocab_size=50, d_model=8, num_heads=2, d_ff=16, max_len=6
        )
        probabilities, cache = model.forward(x_train[:2])
        self.assertEqual(probabilities.shape, (2, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(cache["attn_weights"].shape, (2, 2, 6, 6))
        np.testing.assert_allclose(cache["attn_weights"].sum(axis=-1), 1.0)

        initial_head = model.Wc.copy()
        optimizer = functions["AdamOptimizer"](lr=0.005)
        with contextlib.redirect_stdout(io.StringIO()):
            history = functions["train_model"](
                model, optimizer, x_train, y_train, x_val, y_val,
                epochs=1, batch_size=4,
            )
        self.assertEqual(len(history["train_losses"]), 1)
        self.assertTrue(np.isfinite(history["train_losses"][0]))
        self.assertTrue(np.isfinite(history["val_losses"][0]))
        self.assertFalse(np.array_equal(initial_head, model.Wc))


if __name__ == "__main__":
    unittest.main()
