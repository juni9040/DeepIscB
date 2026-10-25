# DeepIscB

A deep learning model that predicts IscB editing efficiency from a 30-nt target context sequence.

DeepIscB uses a multi-branch 1D CNN over the one-hot encoded sequence, combined with biophysical
features computed from the sequence (GC content, polynucleotide runs, melting temperature, and
RNA folding free energy). Predictions are averaged over a 5-fold ensemble.

## Installation

The conda environment contains every dependency, including PyTorch, Biopython, and the ViennaRNA
Python bindings:

```bash
conda env create -f env.yaml
conda activate deepIscB
```

## Usage

### 1. Prepare the input file

The input is a CSV file with a single column named `context`, holding one 30-nt target
sequence per row:

```
context
GGCCCCGTTGGCCCCCCTGGAAAGAAAGGA
TTAGTGGTGAATTATATAGCAAAGAAATCC
```

See `example_input.csv` for a complete example. All sequences must be exactly 30 nt long and
contain only A, T, G, and C.

### 2. Run the prediction

The input and output paths are set near the top of `DeepIscB_test.py`:

```python
input_file  = 'example_input.csv'
output_file = 'example_output.csv'
```

Edit those two lines to point at your own files, then run:

```bash
python DeepIscB_test.py
```

The script loads the five trained models from `model/`, so run it from the repository root.
It uses a GPU when one is available and falls back to CPU otherwise.

### 3. Read the output

The output CSV holds the input sequence alongside the ensemble prediction:

| Column | Description |
| --- | --- |
| `context` | The input target sequence |
| `enIscB_score` | Predicted editing efficiency for enIscB, in percent |
| `enOgeuIscB_score` | Predicted editing efficiency for enOgeuIscB, in percent |

See `example_output.csv` for a complete example.

## Training

`DeepIscB_train.py` trains the model from scratch. It expects an input CSV that also carries the
measured `en`, `enO`, and `LFC` columns, and it writes one model file per fold into `model/`.
The hyperparameters of the released models are recorded in
`20261024_feature_fix_FM1test_params.json`.

## Repository contents

| Path | Description |
| --- | --- |
| `DeepIscB_test.py` | Prediction script |
| `DeepIscB_train.py` | Training script |
| `model/` | Five trained fold models |
| `env.yaml` | Conda environment specification |
| `example_input.csv`, `example_output.csv` | Minimal input and output examples |
| `0521_IscB_input_FM1test.csv` | Full dataset used for training |
| `20261024_feature_fix_FM1test_IscB_output.csv` | Predictions on the full dataset |
| `20261024_feature_fix_FM1test_params.json` | Hyperparameters of the released models |
