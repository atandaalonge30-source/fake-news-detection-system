import argparse

import pandas as pd

from src.train import data_profile, load_data


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


def test_load_data_removes_normalized_duplicates_before_splitting(tmp_path):
    data_path = tmp_path / "news.csv"
    pd.DataFrame({
        "text": [
            "A fake article with enough content to be used as a training example.",
            "A fake article with enough content to be used as a training example!",
            "A real article with enough content to be used as a training example.",
        ],
        "label": ["fake", "fake", "real"],
    }).to_csv(data_path, index=False)
    args = argparse.Namespace(data=data_path, fake_csv=None, true_csv=None, min_characters=1)

    result = load_data(args)

    assert len(result) == 2
    assert result.attrs["deduplication"] == {
        "rows_before_deduplication": 3,
        "duplicate_texts_removed": 1,
        "conflicting_duplicate_groups_removed": 0,
        "conflicting_duplicate_rows_removed": 0,
    }


def test_load_data_discards_all_copies_with_conflicting_labels(tmp_path):
    data_path = tmp_path / "news.csv"
    pd.DataFrame({
        "text": [
            "One repeated article with a sufficiently long text for testing.",
            "One repeated article with a sufficiently long text for testing!",
            "A unique real article with sufficient text for the training sample.",
            "A unique fake article with sufficient text for the training sample.",
        ],
        "label": ["fake", "real", "real", "fake"],
    }).to_csv(data_path, index=False)
    args = argparse.Namespace(data=data_path, fake_csv=None, true_csv=None, min_characters=1)

    result = load_data(args)

    assert len(result) == 2
    assert not result["text"].str.startswith("One repeated").any()
    assert result.attrs["deduplication"]["conflicting_duplicate_groups_removed"] == 1
    assert result.attrs["deduplication"]["conflicting_duplicate_rows_removed"] == 2


def test_data_profile_summarizes_class_counts_and_duplicate_removal():
    data = pd.DataFrame({
        "text": ["A repeated article", "A repeated article", "Another story"],
        "label": ["fake", "fake", "real"],
    })

    profile = data_profile(data, "news.csv")

    assert profile["source"] == "news.csv"
    assert profile["rows"] == 3
    assert profile["class_counts"] == {"fake": 2, "real": 1}
    assert profile["class_percentages"] == {"fake": 66.7, "real": 33.3}
    assert profile["duplicate_texts_removed"] == 1
    assert profile["dataset_attribution"]["license"] == "Check the source dataset's license"
