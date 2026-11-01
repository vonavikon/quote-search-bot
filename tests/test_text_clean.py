"""Display typography: guillemet quotes, the inter-word dash,
everything else as is."""
import text_clean


def test_quotes_paired():
    assert text_clean.apply('он сказал "да"') == "он сказал «да»"


def test_quotes_multiline_span():
    """A multi-line straight quote: opened on one line, closed on
    another."""
    out = text_clean.apply('"строка первая\nстрока вторая"')
    assert out == "«строка первая\nстрока вторая»"


def test_trailing_spaces_stripped():
    assert text_clean.apply("хвост  \nпробелы \t\nчисто") == "хвост\nпробелы\nчисто"


def test_dash_between_words():
    assert text_clean.apply("жизнь - это боль") == "жизнь — это боль"


def test_hyphen_inside_word_untouched():
    assert text_clean.apply("кто-то где-то") == "кто-то где-то"


def test_no_changes_needed():
    assert text_clean.apply("«Уже» красиво — да") == "«Уже» красиво — да"
