import argparse

import pandas as pd

from src.train import load_data


def test_load_data_maps_target_labels_as_documented(tmp_path):
    data_path = tmp_path / "news.csv"
    pd.DataFrame({
        "title": ["A real headline", "A fake headline"],
        "content": ["A sufficiently long real article for training.", "A sufficiently long fake article for training."],
        "target": [0, 1],
    }).to_csv(data_path, index=False)
    args = argparse.Namespace(data=data_path, fake_csv=None, true_csv=None, min_characters=1)

    result = load_data(args)

    assert result.set_index("text")["label"].to_dict() == {
        "A real headline A sufficiently long real article for training.": "real",
        "A fake headline A sufficiently long fake article for training.": "fake",
    }
