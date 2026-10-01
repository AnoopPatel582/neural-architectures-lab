# Neural Architectures Lab

Four Jupyter notebooks exploring neural network architectures for image classification and sentiment analysis. They combine implementations of core operations from scratch with PyTorch experiments and model comparisons.

## Notebooks

| Notebook | What it contains |
| --- | --- |
| [01 · CNN from scratch and training](01_CNN_From_Scratch_and_Training.ipynb) | NumPy convolution, pooling, activation, flattening, and MLP components; a PyTorch CNN experiment on CIFAR-10; feature visualizations and backpropagation derivations. |
| [02 · RNN and LSTM sentiment](02_RNN_LSTM_Sentiment.ipynb) | Vanilla RNN and LSTM implementations, backpropagation through time, IMDb sentiment experiments, and comparisons of gates, optimizers, and sequence lengths. |
| [03 · Transformer encoder from scratch](03_Transformer_Encoder_From_Scratch.ipynb) | NumPy token representations, positional encoding, scaled dot-product attention, multi-head attention, encoder blocks, and IMDb sentiment classification. |
| [04 · Long-context sequence models](04_Long_Context_Sequence_Models.ipynb) | PyTorch comparisons of LSTM, Transformer, linear state space, and selective state space models using IMDb reviews and WikiText-2 text. |

The notebooks are independent. Their numbered names reflect the topics found in the files; the source repository's filenames did not match several notebook contents.

## Run locally

Install Python, create an environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

Open a notebook and run its cells from top to bottom. The experiments download CIFAR-10, IMDb, or WikiText-2 when needed. Training and dataset downloads can take substantial time and may benefit from a GPU. The notebooks include saved outputs from the source project; their full training runs have **not** been independently reproduced in this adaptation.

## Dependencies

The notebooks use NumPy, Pandas, Matplotlib, scikit-learn, PyTorch, torchvision, TensorFlow/Keras, and Hugging Face Datasets. See [requirements.txt](requirements.txt). Library and hardware compatibility should be checked in the environment where the notebooks are run.

## Project origin

This repository is an independent presentation of [Shreyansh912/Machine_Learning](https://github.com/Shreyansh912/Machine_Learning), adapted with permission. The initial adaptation gives the repository an accurate name, corrected notebook filenames, a clean Git history, dependency list, and documentation. The notebook implementations and results remain available for a later improvement and reproducibility review.
