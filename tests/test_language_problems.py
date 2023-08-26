from server.utils.language_utils import fix_lang_text_problems


def test_fix_problems():
    assert fix_lang_text_problems("Лиговский проспект", "../server/utils/dict.txt") == "Лиговский проспект"
    assert fix_lang_text_problems("Ligovskiy prospekt", "../server/utils/dict.txt") == "Лиговский проспект"
    assert fix_lang_text_problems("Kbujdcrbq ghjcgtrn", "../server/utils/dict.txt") == "Лиговский проспект"
