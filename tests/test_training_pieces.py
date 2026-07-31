"""The pieces run 2 prescribed: augmentation, multi-class, confusables, pooling.

These went in as a batch and shipped untested, which is how a wrong number got
published once already. Each test pins the property the piece exists for, not
merely that it runs.
"""

import numpy as np
import pytest
import torch

from o1sound import O1Sound, O1SoundConfig
from o1sound.data.augment import (WaveformAugment, add_noise, spec_augment,
                                  speed_perturb, time_shift)
from o1sound.data.mswc import KeywordSpec, _edit_distance


class TestAugmentationIsLabelPreserving:
    """Augmentation must change the recording without changing the word."""

    def test_output_shape_and_range_are_stable(self):
        aug = WaveformAugment(seed=0, p=1.0)
        x = (np.random.randn(16000) * 0.1).astype(np.float32)
        for _ in range(20):
            y = aug(x)
            assert y.shape == x.shape
            assert np.isfinite(y).all()
            assert np.abs(y).max() <= 1.0    # clipped, never overflows

    def test_time_shift_preserves_energy_it_keeps(self):
        x = np.ones(100, dtype=np.float32)
        import random
        y = time_shift(x, 0.2, random.Random(0))
        assert y.shape == x.shape
        assert 80 <= int(y.sum()) <= 100     # only the shifted-out tail is lost

    def test_noise_lowers_snr_monotonically(self):
        import random
        x = np.sin(np.linspace(0, 50, 16000)).astype(np.float32)
        loud = add_noise(x, (0.0, 0.0), random.Random(0))
        quiet = add_noise(x, (30.0, 30.0), random.Random(0))
        assert np.mean((loud - x) ** 2) > np.mean((quiet - x) ** 2)

    def test_speed_perturb_refits_the_window(self):
        import random
        x = (np.random.randn(16000) * 0.1).astype(np.float32)
        for rate in (0.9, 1.1):
            y = speed_perturb(x, (rate, rate), random.Random(0))
            assert y.shape == x.shape

    def test_identity_when_probability_is_zero(self):
        """p=0 must be a true no-op -- a silent transform would corrupt dev."""
        aug = WaveformAugment(seed=0, p=0.0)
        x = (np.random.randn(1000) * 0.1).astype(np.float32)
        assert np.array_equal(aug(x), x)


class TestSpecAugment:
    def test_masks_without_changing_shape(self):
        mel = torch.randn(4, 60, 40)
        out = spec_augment(mel, freq_width=6, time_width=12)
        assert out.shape == mel.shape
        assert not torch.equal(out, mel)

    def test_leaves_the_input_untouched(self):
        mel = torch.randn(2, 30, 40)
        before = mel.clone()
        spec_augment(mel)
        assert torch.equal(mel, before)


class TestConfusableSelection:
    def test_edit_distance_ranks_near_misses_above_unrelated_words(self):
        """The whole point: hollow must sort closer to hello than seven does."""
        assert _edit_distance("hello", "hollow") < _edit_distance("hello", "seven")
        assert _edit_distance("hello", "hell") < _edit_distance("hello", "apple")
        assert _edit_distance("hello", "hello") == 0

    def test_works_in_non_latin_script(self):
        """MSWC keys words natively, so this runs on Cyrillic and Arabic too."""
        assert _edit_distance("привет", "привит") == 1
        assert _edit_distance("سلام", "سلام") == 0

    def test_spec_defaults_to_uniform_negatives(self):
        assert KeywordSpec({"en": "hello"}).confusable_frac == 0.0


class TestPooling:
    @pytest.mark.parametrize("mode", ["mean", "max", "attn"])
    def test_every_mode_produces_finite_logits(self, mode):
        m = O1Sound(O1SoundConfig(hidden=32, n_layers=1, n_classes=2, pooling=mode)).eval()
        with torch.no_grad():
            out = m(torch.randn(3, 25, 40))
        assert out.shape == (3, 2)
        assert torch.isfinite(out).all()

    def test_mean_parameter_set_is_unchanged(self):
        """Adding pooling modes must not perturb existing checkpoints."""
        a = O1Sound(O1SoundConfig(hidden=32, n_layers=1, n_classes=2))
        b = O1Sound(O1SoundConfig(hidden=32, n_layers=1, n_classes=2, pooling="mean"))
        assert set(a.state_dict()) == set(b.state_dict())
        assert "pool_score.weight" not in a.state_dict()

    def test_attn_adds_exactly_one_scorer(self):
        base = O1Sound(O1SoundConfig(hidden=32, n_layers=1, n_classes=2)).num_parameters()
        attn = O1Sound(O1SoundConfig(hidden=32, n_layers=1, n_classes=2,
                                     pooling="attn")).num_parameters()
        assert attn - base == 32 + 1          # one Linear(hidden, 1)

    def test_max_pooling_survives_a_diluting_window(self):
        """Mean pooling's failure mode, made concrete: one distinctive frame in
        a long quiet window should still register.

        The frame has to be spectrally SHAPED, not merely loud. The model
        LayerNorms across mel bands, so a uniformly-bright frame normalises to
        exactly the same vector as silence -- loudness carries no information,
        only the shape across bands does.
        """
        torch.manual_seed(0)
        m = O1Sound(O1SoundConfig(hidden=16, n_layers=1, n_classes=2, pooling="max")).eval()
        quiet = torch.zeros(1, 100, 40)
        spike = quiet.clone()
        spike[0, 50] = torch.linspace(-3.0, 3.0, 40)   # shaped, not flat
        with torch.no_grad():
            d = (m(spike) - m(quiet)).abs().max()
        assert d > 1e-4

    def test_a_uniformly_loud_frame_is_invisible_after_layernorm(self):
        """Pins the property the test above had to work around: per-frame
        LayerNorm over mel bands discards absolute level by construction."""
        m = O1Sound(O1SoundConfig(hidden=16, n_layers=1, n_classes=2)).eval()
        quiet = torch.zeros(1, 20, 40)
        loud = torch.full((1, 20, 40), 5.0)
        with torch.no_grad():
            assert torch.allclose(m(quiet), m(loud), atol=1e-5)

    def test_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError, match="pooling"):
            O1SoundConfig(pooling="median")


class TestMultiClassHead:
    def test_class_count_flows_into_the_head(self):
        m = O1Sound(O1SoundConfig(hidden=32, n_layers=1, n_classes=18))
        with torch.no_grad():
            out = m(torch.randn(2, 20, 40))
        assert out.shape == (2, 18)

    def test_eighteen_classes_stay_inside_the_budget(self):
        """17 greetings + other, at the shipping width, must still fit 7 MB."""
        m = O1Sound(O1SoundConfig(hidden=640, n_layers=2, n_classes=18, pooling="attn"))
        assert m.num_parameters() * 4 / 1e6 < 7.0

    def test_streaming_step_matches_the_batched_path_multiclass(self):
        """The property the export depends on must hold for N classes too."""
        torch.manual_seed(0)
        m = O1Sound(O1SoundConfig(hidden=24, n_layers=2, n_classes=6)).eval()
        mel = torch.randn(2, 20, 40)
        with torch.no_grad():
            h = m.norm(mel)
            for cell in m.cells:
                h = cell(h)
            batched_last = m.head(h[:, -1])
            state = m.init_state(2)
            for t in range(mel.shape[1]):
                streamed, state = m.step(mel[:, t], state)
        assert torch.allclose(batched_last, streamed, atol=1e-5)
