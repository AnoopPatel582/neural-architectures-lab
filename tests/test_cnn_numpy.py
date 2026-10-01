"""Fast checks for notebook syntax and the NumPy CNN operations."""

import ast
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CNN_NOTEBOOK = ROOT / "01_CNN_From_Scratch_and_Training.ipynb"


def load_notebook_functions(path: Path) -> dict:
    """Load function definitions from the notebook without running training cells."""
    namespace = {"np": np}
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if functions:
            module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
            exec(compile(module, str(path), "exec"), namespace)
    return namespace


class NotebookChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_notebook_functions(CNN_NOTEBOOK)

    def test_all_code_cells_have_valid_python_syntax(self):
        for path in sorted(ROOT.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell_number, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "code":
                    with self.subTest(notebook=path.name, cell=cell_number):
                        ast.parse("".join(cell.get("source", [])))

    def test_convolution_known_values(self):
        image = np.arange(1, 10, dtype=float).reshape(3, 3, 1)
        kernel = np.ones((2, 2, 1), dtype=float)
        actual = self.functions["convolution_function"](
            image, kernel, activation=None
        )
        np.testing.assert_array_equal(actual, [[12, 16], [24, 28]])

    def test_convolution_padding_and_channel_validation(self):
        image = np.arange(1, 10, dtype=float).reshape(3, 3, 1)
        kernel = np.ones((3, 3, 1), dtype=float)
        actual = self.functions["convolution_function"](
            image, kernel, padding=1, activation=None
        )
        self.assertEqual(actual.shape, (3, 3))
        self.assertEqual(actual[0, 0], 12)
        self.assertEqual(actual[1, 1], 45)
        with self.assertRaises(AssertionError):
            self.functions["convolution_function"](
                image, np.ones((2, 2, 2)), activation=None
            )

    def test_max_and_average_pooling(self):
        feature_map = np.arange(1, 17, dtype=float).reshape(4, 4)
        pool = self.functions["pooling_function"]
        np.testing.assert_array_equal(pool(feature_map, mode="max"), [[6, 8], [14, 16]])
        np.testing.assert_array_equal(
            pool(feature_map, mode="avg"), [[3.5, 5.5], [11.5, 13.5]]
        )

    def test_full_forward_pass_shapes_and_probabilities(self):
        np.random.seed(42)
        result = self.functions["cnn_feedforward"](np.zeros((32, 32, 3)))
        self.assertEqual(result["conv1"].shape, (32, 32, 4))
        self.assertEqual(result["pool1"].shape, (16, 16, 4))
        self.assertEqual(result["conv2"].shape, (16, 16, 4))
        self.assertEqual(result["pool2"].shape, (8, 8, 4))
        self.assertEqual(result["flat_output"].shape, (256,))
        probabilities = result["softmax_output"]
        self.assertEqual(probabilities.shape, (10,))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
