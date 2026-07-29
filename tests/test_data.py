import numpy as np
import pytest
from o1sound.data import KeywordSpec, MSWCWakeWord, fit_length
from o1sound.keywords import GREETINGS


def test_fit_length_pads_and_crops():
    x = np.arange(10, dtype=np.float32)
    assert fit_length(x, 16).shape == (16,)
    assert fit_length(x, 4).shape == (4,)
    assert fit_length(x, 10) is x


def test_missing_root_is_actionable():
    with pytest.raises(FileNotFoundError, match="fetch_mswc"):
        MSWCWakeWord("./definitely-not-here", KeywordSpec(GREETINGS))


def test_greetings_are_lowercase_single_tokens():
    """MSWC keys clips by word directory; a space or capital never matches."""
    for lang, word in GREETINGS.items():
        assert word == word.lower(), lang
        assert " " not in word, lang
