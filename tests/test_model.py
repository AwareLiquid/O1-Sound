import torch
from o1sound import LogMel, O1Sound, O1SoundConfig


def test_logmel_shape_and_frame_count():
    fe = LogMel()
    mel = fe(torch.randn(3, 16000))
    assert mel.shape == (3, fe.num_frames(16000), 40)
    assert torch.isfinite(mel).all()


def test_streaming_matches_batched():
    """step() and forward() must be the same recurrence.

    If these ever diverge, the exported streaming graph stops matching what was
    trained -- the failure would be silent and only show up as accuracy loss on
    device.
    """
    torch.manual_seed(0)
    m = O1Sound(O1SoundConfig(hidden=32, n_layers=2, n_classes=2)).eval()
    mel = torch.randn(2, 25, 40)
    with torch.no_grad():
        h = m.norm(mel)
        for cell in m.cells:
            h = cell(h)
        batched_last = m.head(h[:, -1])

        state = m.init_state(2)
        for t in range(mel.shape[1]):
            streamed, state = m.step(mel[:, t], state)
    assert torch.allclose(batched_last, streamed, atol=1e-5)


def test_state_is_constant_size():
    m = O1Sound(O1SoundConfig(hidden=64, n_layers=2, n_classes=2)).eval()
    state = m.init_state(1)
    before = sum(s.numel() for s in state)
    with torch.no_grad():
        for _ in range(500):
            _, state = m.step(torch.randn(1, 40), state)
    assert sum(s.numel() for s in state) == before
    assert m.state_bytes(1) == before * 4


def test_tau_spans_the_configured_range():
    cfg = O1SoundConfig(hidden=128, n_layers=1, tau_min=1.0, tau_max=24.0)
    tau = 1.0 / O1Sound(cfg).cells[0].alpha()
    assert torch.isclose(tau.min(), torch.tensor(1.0), atol=0.05)
    assert torch.isclose(tau.max(), torch.tensor(24.0), atol=0.5)
    assert (tau > 0).all()


def test_config_rejects_bad_tau():
    for kw in ({"tau_min": 0.0}, {"tau_min": 30.0, "tau_max": 10.0}):
        try:
            O1SoundConfig(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")
