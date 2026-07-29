"""Greeting words per language, as they appear in MSWC's word directories.

MSWC keys clips by the written form, so these are the orthographic entries to
look for -- not transliterations. A language belongs here only once its
directory has been confirmed to exist in a downloaded MSWC subset; the loader
reports any that are missing rather than quietly training on fewer languages
than the spec claims.
"""

GREETINGS: dict[str, str] = {
    "en": "hello",
    "de": "hallo",
    "nl": "hallo",
    "fr": "bonjour",
    "es": "hola",
    "it": "ciao",
    "pt": "ola",
    "ca": "hola",
    "pl": "czesc",
    "cs": "ahoj",
    "ru": "privet",
    "uk": "pryvit",
    "tr": "merhaba",
    "id": "halo",
    "vi": "chao",
    "sv-SE": "hej",
    "ro": "salut",
    "el": "geia",
    "fa": "salam",
    "ar": "marhaba",
}
