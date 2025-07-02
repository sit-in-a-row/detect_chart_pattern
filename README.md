# Detect Chart Pattern

This repository contains tools and data for detecting classical chart patterns from OHLCV time series.

## Repository structure

- `algo_dataset/` – Automatically generated CSV datasets for six chart patterns.
- `create_dataset/` – Python package for creating synthetic chart pattern datasets.
- `backtest/` – Notebook demonstrating model predictions on real data.
- `train/` – Notebook for training a CNN‑LSTM classifier.
- `save_model/` – Pretrained model (`chart_pattern_model.h5`).
- `notepad.txt` – Notes about external Kaggle dataset and pattern codes.

## Supported patterns

1. Ascending Triangle
2. Ascending Wedge
3. Descending Triangle
4. Descending Wedge
5. Double Top
6. Double Bottom

## Dataset generation

Synthetic datasets are produced using the utilities in `create_dataset`. Example usage:

```python
from create_dataset import create_dataset
create_dataset(generation_count=1000, n_min=50, n_max=120)
```

This generates random OHLCV sequences for each pattern type and saves them under `algo_dataset/`.

## Training

The notebook `train/train.ipynb` loads the generated CSV files, constructs sliding windows and trains a CNN‑LSTM model to classify the six patterns. The resulting model can be found in `save_model/chart_pattern_model.h5`.

Required libraries include:

- TensorFlow / Keras
- pandas
- numpy
- scikit-learn
- mplfinance

## Backtesting

`backtest/backtest.ipynb` shows how to load the trained model and run predictions on OHLCV data fetched from `pykrx`. The notebook visualizes each segment with the predicted pattern and probability.

## Notes

The repository includes over 600 MB of synthetic CSV data under `algo_dataset`. Original Kaggle data references are listed in `notepad.txt`.

