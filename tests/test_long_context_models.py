"""Offline training checks for all four long-context model classes."""

import ast
import json
import math
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


NOTEBOOK = Path(__file__).resolve().parents[1] / "04_Long_Context_Sequence_Models.ipynb"


def load_definitions():
    namespace = {
        "math": math, "np": np, "time": time, "torch": torch,
        "nn": nn, "Dataset": Dataset, "DataLoader": DataLoader,
    }
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


class LongContextChecks(unittest.TestCase):
    def test_all_models_train_and_evaluate_on_short_sequences(self):
        functions = load_definitions()
        rng = np.random.default_rng(42)
        examples = [
            (rng.integers(1, 50, size=16).tolist(), index % 2)
            for index in range(8)
        ]
        loader = DataLoader(functions["TextDataset"](examples), batch_size=4)
        models = (
            ("LSTM", functions["LSTMClassifier"](50, 16, 16, 2)),
            ("Transformer", functions["TransformerClassifier"](50, 16, 4, 32, 1, 2, max_len=16)),
            ("LinearSSM", functions["EmbeddedLinearSSM"](50, 16, 16, 2)),
            ("SelectiveSSM", functions["EmbeddedSelectiveSSM"](50, 16, 16, 2)),
        )
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")
        for name, model in models:
            with self.subTest(model=name):
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                train_loss, train_acc, _ = functions["train_one_epoch"](
                    model, loader, optimizer, criterion, device
                )
                test_loss, test_acc = functions["evaluate"](
                    model, loader, criterion, device
                )
                self.assertTrue(np.isfinite([train_loss, train_acc, test_loss, test_acc]).all())


if __name__ == "__main__":
    unittest.main()
