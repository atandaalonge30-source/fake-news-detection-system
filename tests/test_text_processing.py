from src.text_processing import clean_text


def test_clean_text_normalizes_url_html_and_case():
    assert clean_text("HELLO <b>World</b>! Visit https://example.com") == "hello world visit url"


def test_clean_text_handles_missing_value():
    assert clean_text(None) == ""
