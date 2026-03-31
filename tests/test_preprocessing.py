from src.data.preprocessing import clean_text


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("Hello World") == "hello world"

    def test_remove_punctuation(self):
        assert clean_text("Hello, world!") == "hello world"

    def test_remove_digits(self):
        assert clean_text("test123") == "test"

    def test_normalize_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_non_string_input(self):
        assert isinstance(clean_text(123), str)
        assert isinstance(clean_text(None), str)

    def test_cyrillic_preserved(self):
        result = clean_text("Привет мир!")
        assert "привет" in result
        assert "мир" in result

    def test_empty_string(self):
        assert clean_text("") == ""
