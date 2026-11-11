# WearFed

Core implementation of WearFed for federated learning on wearable devices, provided for code inspection during peer review.

## Current release status

This snapshot contains the supplied MotionSense and UCI-HAR implementations. It is **not yet a complete runnable reproduction package**. The following source files were absent from the supplied archive:

- `motionsense/motionsense_dataset.py`
- `uci_har/uci_har_dataset.py`
- The original UCI-HAR `server.py` source. The included `uci_har/server.py` is a loader for the supplied CPython bytecode, not the original implementation source.

Restore the original dataset loaders and UCI-HAR server source before using this repository as a complete reproduction package. The missing preprocessing and server implementations have not been reconstructed or substituted.

## Repository structure

```text
.
├── motionsense/
│   ├── clients.py
│   ├── main.py
│   ├── model.py
│   └── server.py
├── uci_har/
│   ├── bytecode/
│   │   ├── server.cpython-312.pyc
│   │   └── server.cpython-39.pyc
│   ├── clients.py
│   ├── main.py
│   ├── model.py
│   └── server.py
├── .gitignore
├── README.md
└── requirements.txt
```

- `clients.py`: local training, device state and energy modeling, and model configuration selection.
- `model.py`: model architectures, elastic model configurations, and quantization utilities.
- `server.py`: federated training, client selection, model aggregation, and prototype aggregation. For UCI-HAR, see the bytecode limitation above.
- `main.py`: dataset-specific training entry point and experiment settings.

## Environment

```bash
python -m pip install -r requirements.txt
```

The supplied UCI-HAR server bytecode is specific to CPython 3.9 or 3.12. The loader selects the file matching the interpreter. Dependency versions were not supplied and are not pinned in this snapshot. End-to-end training has not been verified for this release.

## Data and execution

First restore the missing source files listed above and install the dependencies. Run the following commands from the repository root:

```bash
python motionsense/main.py
python uci_har/main.py
```

These commands cannot run successfully with the current incomplete snapshot.

The current entry points expect local data at:

- MotionSense: `./data`, containing `A_DeviceMotion_data`. The entry point requests windows of 128 samples, a stride of 64, and 12 input features. The original missing loader is needed to establish the preprocessing and split.
- UCI-HAR: `./data/UCI-HAR Dataset`, containing `train/X_train.txt`, `train/y_train.txt`, `test/X_test.txt`, and `test/y_test.txt`.

Training parameters, including client counts, partition settings, random seeds, and round counts, are specified in each `main.py`.

## Included scope

This repository contains the supplied core method code. Baseline implementations, experimental result files, plots, logs, trained model weights, raw datasets, and preprocessed datasets are not included. Model aggregation by sample count is part of the training implementation and is retained.

Only the implementations listed above are included. This snapshot does not establish that every experiment in the associated manuscript can be reproduced.
