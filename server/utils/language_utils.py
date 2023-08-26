import os
import re
import string

en_to_ru_layout_dict = dict(zip(map(ord, "qwertyuiop[]asdfghjkl;'zxcvbnm,./`"
                                         'QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?~'),
                                "йцукенгшщзхъфывапролджэячсмитьбю.ё"
                                'ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,Ё'))


def transform_layout(text: str, dictionary=en_to_ru_layout_dict):
    return text.translate(dictionary)


def check_words_in_dict_or_startswith(text: str, addresses_dict):
    for word in text.translate(str.maketrans('', '', string.punctuation)).split(" "):
        if word in addresses_dict:
            return True

        for dict_word in addresses_dict:
            if dict_word.startswith(word):
                return True

    return False


def detect_keyboard_mismatch(text: str, addresses_dict):
    if len(text) <= 3 or re.compile('[а-яА-Я]').search(text):
        return False

    text = transform_layout(text.lower())

    return check_words_in_dict_or_startswith(text, addresses_dict)


def transliterate_to_russian(text: str):
    translit_dict = {
        'a': 'а', 'b': 'б', 'c': 'ц', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
        'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
        'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т',
        'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'й', 'z': 'з',
        'A': 'А', 'B': 'Б', 'C': 'Ц', 'D': 'Д', 'E': 'Е', 'F': 'Ф', 'G': 'Г',
        'H': 'Х', 'I': 'И', 'J': 'Й', 'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н',
        'O': 'О', 'P': 'П', 'Q': 'К', 'R': 'Р', 'S': 'С', 'T': 'Т',
        'U': 'У', 'V': 'В', 'W': 'В', 'X': 'КС', 'Y': 'Й', 'Z': 'З',
        ' ': ' ',
    }

    russian_text = ''
    for char in text:
        if char in translit_dict:
            russian_text += translit_dict[char]
        else:
            russian_text += char

    return russian_text


def detect_transliteration(text: str, addresses_dict):
    if len(text) <= 3 or re.compile('[а-яА-Я]').search(text):
        return False

    text = transliterate_to_russian(text.lower())

    return check_words_in_dict_or_startswith(text, addresses_dict)


def fix_lang_text_problems(text: str, dict_path="dict.txt"):
    addresses_dict = set(list(map(lambda x: x.replace("\n", ""),
                                  open(os.path.abspath(dict_path), "r", encoding="u8").readlines())))

    if detect_transliteration(text, addresses_dict):
        return transliterate_to_russian(text)

    if detect_keyboard_mismatch(text, addresses_dict):
        return transform_layout(text)

    return text
