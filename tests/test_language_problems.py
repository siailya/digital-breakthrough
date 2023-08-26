from utils.language_detection import fix_lang_text_problems


def test_fix_problems():
    assert fix_lang_text_problems("Лиговский проспект", "../utils/dict.txt") == "Лиговский проспект"
    assert fix_lang_text_problems("Ligovskiy prospekt", "../utils/dict.txt") == "Лиговский проспект"
    assert fix_lang_text_problems("Kbujdcrbq ghjcgtrn", "../utils/dict.txt") == "Лиговский проспект"
