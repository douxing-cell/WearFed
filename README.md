# WearFed

Implementation of WearFed and comparison methods for federated learning on
UCI-HAR and MotionSense.

## Code structure

- `uci_har/`: UCI-HAR model, client logic, server, dataset loader, and training entry points.
- `motionsense/`: MotionSense model, client logic, server, dataset loader, and training entry points.
- `clients.py`: local training, device states, energy modeling, and model configuration selection.
- `model.py`: model architectures, elastic model configurations, and quantization utilities.
- `server.py`: federated training, client selection, parameter aggregation, and prototype aggregation.
- `requirements.txt`: Python dependencies.

## Environment

Create a Python environment and install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The UCI-HAR bytecode loader selects the corresponding implementation for CPython
3.9 or 3.12.

## Datasets

Obtain the datasets from their original publishers:

- UCI-HAR: https://doi.org/10.24432/C54S4K
- MotionSense: https://github.com/mmalekzadeh/motion-sense

Place UCI-HAR under `uci_har/data/UCI-HAR Dataset/`, containing:

```text
train/X_train.txt
train/y_train.txt
test/X_test.txt
test/y_test.txt
```

The MotionSense loader reads activity CSV files from
`motionsense/data/A_DeviceMotion_data/A_DeviceMotion_data/`.
The dataset root can be configured in the corresponding training entry point.

## Experiment configuration

Set the data path, Dirichlet alpha, random seed, client counts, round count, and
method parameters in the entry point for the selected dataset and method.
The parameter values used to construct the model, partition the data, and run
training are specified explicitly in each script. Select the settings for the
experiment before starting training.

## Training entry points

Run each script from its dataset directory so relative data paths resolve there.

| Method | UCI-HAR | MotionSense |
| --- | --- | --- |
| WearFed | `main.py` | `main.py` |
| FedAvg | `main_fedavg.py` | `main_fedavg.py` |
| FedProx | `main_fedprox.py` | `main_fedprox.py` |
| REWAFL | `main_rewafl.py` | `main_rewafl.py` |
| Dyn-Pruning | `main_dyn_prune.py` | `main_static_pruning.py` |
| EFLCF | `main_eflcf.py` | `main_eflcf.py` |

For UCI-HAR:

```bash
cd uci_har
python main.py
```

For MotionSense, open a terminal at the repository root and run:

```bash
cd motionsense
python main.py
```

Use the corresponding filename from the table to run a comparison method.
Training logs and other outputs are written locally by the selected script.

## Citation

When using this implementation in research, please cite the associated WearFed paper.
