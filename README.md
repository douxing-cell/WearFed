# WearFed

Core implementation of WearFed for federated learning on wearable devices.

## Code structure

```text
.
├── motionsense/
│   ├── clients.py
│   ├── main.py
│   ├── model.py
│   └── server.py
├── uci_har/
│   ├── bytecode/
│   ├── clients.py
│   ├── main.py
│   ├── model.py
│   └── server.py
├── .gitignore
├── README.md
└── requirements.txt
```

- `clients.py`: client-side local training, device states, energy modeling, and model configuration selection.
- `model.py`: model architectures, elastic model configurations, and quantization utilities.
- `server.py`: federated training, client selection, model aggregation, and prototype aggregation.
- `main.py`: dataset-specific training entry points and experiment settings.

## Environment

```bash
python -m pip install -r requirements.txt
```

The UCI-HAR bytecode loader selects the corresponding implementation for CPython 3.9 or 3.12.

## Data configuration

Configure the datasets and their corresponding loaders in the local project. The training entry points use the following data paths:

- MotionSense: `./data`, containing `A_DeviceMotion_data`.
- UCI-HAR: `./data/UCI-HAR Dataset`, containing `train/X_train.txt`, `train/y_train.txt`, `test/X_test.txt`, and `test/y_test.txt`.

## Training entry points

Run from the repository root after configuring the local environment and data:

```bash
python motionsense/main.py
python uci_har/main.py
```

Training settings, including client counts, data partition parameters, random seeds, and round counts, are specified in each `main.py`.
