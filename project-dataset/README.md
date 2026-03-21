# Chinelotam project-dataset

This dataset supports a freshness classification task with 3 classes:
- `Fresh`
- `Half-Fresh`
- `Spoiled`

## Structure

```
project-dataset/
  train/
    _classes.csv
    fresh/
    half-fresh/
    spoiled/
  valid/
    _classes.csv
    fresh/
    half-fresh/
    spoiled/
```

## CSV format

`_classes.csv` columns:
- `filename`
- `Fresh`
- `Half-Fresh`
- `Spoiled`

One-hot labels, e.g. `1,0,0` for `Fresh`.

## Python dataset loader

File: `dataset_loader.py`

Example usage:

```bash
python dataset_loader.py
```

Dependencies:
- pandas
- Pillow
- torch

If PyTorch isn't needed, you can still use the `summarize_dataset(...)` helper.
