from pathlib import Path

import pandas as pd

from algotrade.data.registry import (
    DatasetSpec,
    default_frame_loader,
    list_datasets,
    load_dataset,
    register_dataset,
    unregister_dataset,
)


def test_register_and_load_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo.csv"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "symbol": ["ABC"] * 3,
            "close": [100, 101, 99],
        }
    )
    frame.to_csv(csv_path, index=False)

    spec = DatasetSpec(
        name="demo",
        path=csv_path,
        frequency="1D",
        loader=default_frame_loader,
    )
    register_dataset(spec)
    try:
        datasets = list(list_datasets())
        assert datasets and datasets[0].name == "demo"
        loaded = load_dataset("demo")
        assert loaded.iloc[0]["symbol"] == "ABC"
    finally:
        unregister_dataset("demo")
