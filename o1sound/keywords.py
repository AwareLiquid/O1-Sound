"""Greeting words per language, exactly as they appear in MSWC's word folders.

Every entry below was VERIFIED against the per-language splits csv
(storage.googleapis.com/public-datasets-mswc/splits/{lang}.tar.gz) on
2026-07-31 -- the number is the clip count found for that exact string. MSWC
keys words in native script with diacritics: the earlier draft of this file
used Latin transliterations (privet, geia, marhaba) and bare ASCII (czesc,
ola, chao), all of which match ZERO folders.

Dropped after verification, not for lack of trying: Turkish (no merhaba/selam/
merhabalar), Arabic (no مرحبا/أهلا), Vietnamese (no chào/chao/xin). A language
with no greeting clips cannot be trained and does not belong in the spec.

Tail warning: languages under ~15 clips (ro 10, id 7, el 6, uk 6) will have
single-digit positives after the train/dev/test split -- their per-language
FRR is directional at best, and the README's rule applies: the multilingual
claim is bounded by the worst measured language.
"""

GREETINGS: dict[str, str] = {
    "en": "hello",        # 301 clips
    "fa": "سلام",         # سلام, 562
    "de": "hallo",        # 121
    "fr": "bonjour",      # 121
    "es": "hola",         # 96
    "pl": "cześć",       # cześć, 40
    "cs": "ahoj",         # 37
    "pt": "olá",          # olá, 24
    "ru": "привет",       # привет, 21
    "sv-SE": "hej",       # 21
    "it": "ciao",         # 18
    "nl": "hallo",        # 15
    "ca": "hola",         # 14
    "ro": "salut",        # 10
    "id": "halo",         # 7
    "el": "γεια",         # γεια, 6
    "uk": "вітаю",        # вітаю, 6 (привіт: 0 clips)
}
