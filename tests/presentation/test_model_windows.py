import pytest

torch = pytest.importorskip("torch")

from tools.presentation.musetalk import feature_windows, positional_encoding


def test_whisper_windows_keep_upstream_time_axis_and_zero_padding():
    states = torch.arange(50, dtype=torch.float32).view(1, 50, 1, 1).expand(1, 50, 5, 384)
    chunks = feature_windows(states, frames=25, fps=25)
    assert chunks.shape == (25, 50, 384)
    # Initial four 50 Hz positions are padding; then real audio begins.
    assert torch.count_nonzero(chunks[0, :20]) == 0
    assert chunks[0, 25, 0] == 1
    assert chunks[1, 25, 0] == 3
    assert torch.isfinite(chunks).all()


def test_audio_shorter_than_video_cannot_be_silently_extended():
    with pytest.raises(ValueError):
        feature_windows(torch.zeros(1, 1, 5, 384), frames=200, fps=25)


def test_positional_encoding_is_finite_and_bounded():
    encoding = positional_encoding(50, 384, torch.bfloat16)
    assert encoding.shape == (50, 384)
    assert torch.isfinite(encoding).all() and encoding.abs().max() <= 1
