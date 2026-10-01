"""Small runtime checks for the NumPy models in the sentiment notebook."""

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / "02_RNN_LSTM_Sentiment.ipynb"


def load_model_definitions():
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
            module = ast.fix_missing_locations(
                ast.Module(body=definitions, type_ignores=[])
            )
            exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


class SentimentModelChecks(unittest.TestCase):
    def test_rnn_and_lstm_complete_a_training_epoch(self):
        functions = load_model_definitions()
        rng = np.random.default_rng(42)
        x_train = rng.integers(1, 32, size=(8, 5))
        y_train = rng.integers(0, 2, size=8)
        x_val = rng.integers(1, 32, size=(4, 5))
        y_val = rng.integers(0, 2, size=4)

        configurations = (
            ("RNN one-hot", functions["VanillaRNN"](32, 4, 4, use_embedding=False), "sgd"),
            ("LSTM", functions["LSTM"](32, 4, 4, use_embedding=True), "adam"),
            ("Coupled LSTM", functions["LSTM"](32, 4, 4, use_embedding=True, coupled=True), "adam"),
        )
        for name, model, optimizer in configurations:
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()):
                history = functions["train_model"](
                    model, x_train, y_train, x_val, y_val,
                    optimizer_name=optimizer, lr=0.001, epochs=1,
                )
                self.assertEqual(len(history["train_loss"]), 1)
                self.assertEqual(len(history["val_loss"]), 1)
                self.assertTrue(np.isfinite(history["train_loss"][0]))
                self.assertTrue(np.isfinite(history["val_loss"][0]))
                self.assertTrue(all(np.isfinite(p).all() for p in model.params.values()))


if __name__ == "__main__":
    unittest.main()
